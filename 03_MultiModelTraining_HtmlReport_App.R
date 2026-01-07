library(shiny)
library(tidyverse)
library(caret)
library(randomForest)
library(e1071)
library(class)
library(rpart)
library(xgboost)
library(Matrix)
library(readr)
library(MLmetrics)
library(htmltools)
library(rmarkdown)
library(knitr)

options(shiny.maxRequestSize = 50 * 1024^2) # hasta 50 MB

ui <- fluidPage(
  titlePanel("Classification / Regression Model Comparison"),
  sidebarLayout(
    sidebarPanel(
      fileInput("datafile", "Upload your TSV File", accept = ".tsv"),
      uiOutput("id_col_ui"),
      uiOutput("col_resp_ui"),
      uiOutput("metrica_ui"),
      sliderInput("cutoff", "Cumulative Gain (%)",
                  min = 0.5, max = 1, value = 0.9, step = 0.05),
      actionButton("go", "Process"),
      tags$hr(),
      downloadButton("download_plot", "Plot PDF"),
      downloadButton("download_metrics", "Metrics TSV"),
      downloadButton("download_importancia", "Importance TSV"),
      downloadButton("download_report", "Download HTML Report"),
      tags$hr(),
      selectInput("model_sel", "Individual Model:",
                  choices = NULL, selected = NULL),
      downloadButton("download_single", "Download Model (.rds)")
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("Preview Data", 
                 dataTableOutput("preview_table")),
        tabPanel("Model Metrics", 
                 plotOutput("acc_plot"),
                 tableOutput("metrics_table"),
                 uiOutput("conf_matrices_ui")),
        tabPanel("Variable Importance",
                 plotOutput("importance_plot"),
                 tableOutput("importance_table"))
      )
    )
  )
)

server <- function(input, output, session) {
  
  datos <- reactive({
    req(input$datafile)
    read_delim(input$datafile$datapath, delim = "\t", col_types = cols(),
               locale = locale(encoding = "UTF-8")) |> drop_na()
  })
  
  output$preview_table <- renderDataTable({
    req(datos())
    head(datos(), 50)
  }, options = list(pageLength = 10, scrollX = TRUE))
  
  output$id_col_ui <- renderUI({
    req(datos())
    choices <- colnames(datos())
    choices_with_none <- c("None" = "", choices)
    selected <- if ("ID" %in% choices) "ID" else ""
    selectInput("id_col", "ID Column Name:", choices = choices_with_none, selected = selected)
  })
  
  output$col_resp_ui <- renderUI({
    req(datos())
    possible_choices <- setdiff(colnames(datos()), input$id_col)
    selectInput("col_resp", "Response Variable:", choices = possible_choices)
  })
  
  output$metrica_ui <- renderUI({
    req(input$col_resp)
    df <- datos()
    if (is.numeric(df[[input$col_resp]])) {
      selectInput("metrica", "Metric:", choices = c("RMSE","Rsquared"), selected="RMSE")
    } else {
      selectInput("metrica", "Metric:", choices = c("Accuracy","Kappa"), selected="Accuracy")
    }
  })
  
  metrics <- reactiveVal()
  importance_full <- reactiveVal()
  models_saved <- reactiveVal()
  conf_matrices <- reactiveVal()
  
  importance_cut <- reactive({
    req(importance_full())
    importance_full() |>
      group_by(class) |>
      arrange(desc(Gain)) |>
      mutate(cum_gain = cumsum(Gain)) |>
      filter(cum_gain <= input$cutoff)
  })
  
  observeEvent(input$go, {
    df <- datos()
    req(input$col_resp)
    
    set.seed(123) 
    
    if (input$id_col != "" && input$id_col %in% names(df)) {
      df <- df |> select(-!!sym(input$id_col))
    }
    
    y <- df[[input$col_resp]]
    
    if (is.numeric(y)) {
      tipo <- "reg"
      ctrl <- trainControl(method = "cv", number = 5)
    } else {
      tipo <- "clas"
      df[[input$col_resp]] <- as.factor(y)
      if (length(levels(df[[input$col_resp]])) > 2) {
        ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE,
                             summaryFunction = multiClassSummary, allowParallel = TRUE,
                             savePredictions = "final")
      } else {
        ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE,
                             summaryFunction = twoClassSummary, allowParallel = TRUE)
      }
    }
    
    predictores_num <- df |> select(where(is.numeric)) |> names()
    predictores_cat <- df |> select(where(~ is.character(.) | is.factor(.))) |> names()
    predictores_num <- setdiff(predictores_num, input$col_resp)
    predictores_cat <- setdiff(predictores_cat, input$col_resp)
    
    df_modelo <- df[, c(predictores_num, predictores_cat, input$col_resp), drop = FALSE]
    idx <- createDataPartition(df_modelo[[input$col_resp]], p = 0.8, list = FALSE)
    train <- df_modelo[idx, ]
    test <- df_modelo[-idx, ]
    
    if (tipo == "clas") {
      modelos <- list(
        rf = train(as.formula(paste(input$col_resp,"~ .")), data = train, method = "rf", trControl = ctrl, metric=input$metrica),
        svm = train(as.formula(paste(input$col_resp,"~ .")), data = train, method = "svmRadial", trControl = ctrl, metric=input$metrica),
        knn = train(as.formula(paste(input$col_resp,"~ .")), data = train, method = "knn", trControl = ctrl, metric=input$metrica),
        cart = train(as.formula(paste(input$col_resp,"~ .")), data = train, method = "rpart", trControl = ctrl, metric=input$metrica),
        xgb = train(as.formula(paste(input$col_resp,"~ .")), data = train, method = "xgbTree", trControl = ctrl, metric=input$metrica)
      )
    } else {
      modelos <- list(
        lm = train(as.formula(paste(input$col_resp,"~ .")), data = train, method = "lm", trControl = ctrl, metric=input$metrica),
        rf = train(as.formula(paste(input$col_resp,"~ .")), data = train, method = "rf", trControl = ctrl, metric=input$metrica),
        tree = train(as.formula(paste(input$col_resp,"~ .")), data = train, method = "treebag", trControl = ctrl, metric=input$metrica),
        xgb = train(as.formula(paste(input$col_resp,"~ .")), data = train, method = "xgbTree", trControl = ctrl, metric=input$metrica)
      )
    }
    
    models_saved(modelos)
    updateSelectInput(session, "model_sel",
                      choices = names(modelos), selected = names(modelos)[1])
    
    preds <- lapply(modelos, predict, newdata = test)
    
    if (tipo == "clas") {
      metrics(map_df(names(preds), \(m){
        cm <- confusionMatrix(preds[[m]], test[[input$col_resp]])
        tibble(Model = m,
               Accuracy = cm$overall["Accuracy"],
               Kappa = cm$overall["Kappa"])
      }))
      
      cm_list <- lapply(names(preds), function(m){
        cm <- confusionMatrix(preds[[m]], test[[input$col_resp]])
        list(model = m, table = as.data.frame(cm$table))
      })
      names(cm_list) <- names(preds)
      conf_matrices(cm_list)
    } else {
      metrics(map_df(names(preds), \(m){
        rmse <- RMSE(preds[[m]], test[[input$col_resp]])
        r2 <- R2(preds[[m]], test[[input$col_resp]])
        tibble(Model = m, RMSE = rmse, Rsquared = r2)
      }))
    }
    
    if (length(c(predictores_num, predictores_cat)) > 0) {
      if (length(predictores_cat) > 0) {
        train_dummies <- dummyVars(" ~ .", data = train[, predictores_cat, drop = FALSE])
        train_dummy_vars <- predict(train_dummies, newdata = train[, predictores_cat, drop = FALSE])
        X <- as.matrix(cbind(train[, predictores_num, drop = FALSE], train_dummy_vars))
      } else {
        X <- as.matrix(train[, predictores_num, drop = FALSE])
      }
      ytrain <- train[[input$col_resp]]
      
      if (tipo == "clas") {
        importance_full(bind_rows(lapply(levels(ytrain), function(cl){
          mdl <- xgboost(
            data = xgb.DMatrix(X, label = ifelse(ytrain == cl, 1, 0)),
            objective = "binary:logistic", nrounds = 50, verbose = 0)
          xgb.importance(model = mdl) |> mutate(class = cl)
        })))
      } else {
        mdl <- xgboost(
          data = xgb.DMatrix(X, label = ytrain),
          objective = "reg:squarederror", nrounds = 50, verbose = 0)
        importance_full(xgb.importance(model = mdl) |> mutate(class = "Regression"))
      }
    }
  })
  
  output$metrics_table <- renderTable({ req(metrics()); metrics() })
  
  output$acc_plot <- renderPlot({
    req(metrics())
    if ("Accuracy" %in% names(metrics())) {
      ggplot(metrics(), aes(Model, Accuracy)) +
        geom_col(fill = "steelblue") + ylim(0,1) + theme_minimal() +
        labs(title = "Model Accuracy Comparison")
    } else {
      ggplot(metrics(), aes(Model, RMSE)) +
        geom_col(fill = "tomato") + theme_minimal() +
        labs(title = "Model RMSE Comparison")
    }
  })
  
  output$importance_table <- renderTable({ req(importance_cut()); importance_cut() })
  
  output$importance_plot <- renderPlot({
    req(importance_cut())
    ggplot(importance_cut(),
           aes(reorder(Feature, Gain), Gain, fill = class)) +
      geom_col(show.legend = FALSE) +
      facet_wrap(~class, scales="free") +
      coord_flip() +
      labs(title = paste0("Variables up to ",
                          input$cutoff*100, "% Cumulative Gain"),
           x = "Variable", y = "Gain") +
      theme_minimal()
  })
  
  output$conf_matrices_ui <- renderUI({
    req(conf_matrices())
    cm_list <- conf_matrices()
    tagList(lapply(names(cm_list), function(m){
      tagList(
        h4(paste("Confusion Matrix:", m)),
        tableOutput(paste0("cm_", m))
      )
    }))
  })
  
  observe({
    req(conf_matrices())
    cm_list <- conf_matrices()
    lapply(names(cm_list), function(m){
      output[[paste0("cm_", m)]] <- renderTable({
        cm_list[[m]]$table
      })
    })
  })
  
  output$download_metrics <- downloadHandler(
    filename = function() paste0("metrics_", Sys.Date(), ".tsv"),
    content = function(file) write_tsv(metrics(), file)
  )
  
  output$download_importancia <- downloadHandler(
    filename = function() paste0("importance_", Sys.Date(), ".tsv"),
    content = function(file) write_tsv(importance_cut(), file)
  )
  
  output$download_plot <- downloadHandler(
    filename = function() paste0("importance_plot_", Sys.Date(), ".pdf"),
    content = function(file){
      pdf(file, 21, 12)
      print(
        ggplot(importance_cut(),
               aes(reorder(Feature, Gain), Gain, fill = class)) +
          geom_col(show.legend = FALSE) +
          facet_wrap(~class, scales="free") +
          coord_flip() +
          labs(title = paste0("Variables up to ",
                              input$cutoff*100, "% Cumulative Gain"),
               x = "Variable", y = "Gain") +
          theme_minimal()
      )
      dev.off()
    }
  )
  
  output$download_single <- downloadHandler(
    filename = function() paste0(input$model_sel,"_",Sys.Date(),".rds"),
    content = function(file){
      saveRDS(models_saved()[[input$model_sel]], file)
    }
  )
  

    # ==== HTML Report Dinámico Funcional ====
    output$download_report <- downloadHandler(
      filename = function() paste0("report_", Sys.Date(), ".html"),
      content = function(file) {
        req(metrics())
        temp_rmd <- tempfile(fileext = ".Rmd")
        
        # Prepara la tabla de importancia con columna "Feature"
        imp <- importance_cut()
        if(!"Feature" %in% names(imp)){
          imp <- imp %>% rename(Feature = rownames(.))
        }
        
        # Convierte matrices de confusión a data.frames legibles
        cm_list <- lapply(conf_matrices(), function(x) x$table)
        
        rmd_lines <- c(
          "---",
          "title: 'Model Comparison Report'",
          "output: html_document",
          "params:",
          "  metrics: NULL",
          "  conf_matrices: NULL",
          "  importance: NULL",
          "  cutoff: NULL",
          "  date: NULL",
          "---",
          "",
          "```{r setup, include=FALSE}",
          "library(ggplot2); library(dplyr); library(knitr)",
          "knitr::opts_chunk$set(echo=FALSE, warning=FALSE, message=FALSE)",
          "```",
          "",
          "# 📊 Model Metrics",
          "",
          "**Date:** `r params$date`  ",
          "",
          "```{r metrics_table}",
          "kable(params$metrics)",
          "```",
          "",
          "# 🧮 Confusion Matrices",
          "```{r conf_matrices}",
          "for(m in names(params$conf_matrices)){",
          "  cat('### Confusion Matrix:', m, '\\n')",
          "  print(kable(params$conf_matrices[[m]]))",
          "}",
          "```",
          "",
          "# 🌟 Variable Importance",
          "",
          "**Top Variables up to `r params$cutoff*100`% cumulative gain**  ",
          "",
          "```{r importance_plot}",
          "ggplot(params$importance, aes(reorder(Feature, Gain), Gain, fill = class)) +",
          "  geom_col(show.legend = FALSE) +",
          "  facet_wrap(~class, scales='free') +",
          "  coord_flip() +",
          "  labs(x='Variable', y='Gain') +",
          "  theme_minimal()",
          "```",
          "",
          "```{r importance_table}",
          "kable(params$importance)",
          "```"
        )
        
        writeLines(rmd_lines, temp_rmd)
        
        rmarkdown::render(
          input = temp_rmd,
          output_file = file,
          params = list(
            metrics = metrics(),
            conf_matrices = cm_list,
            importance = imp,
            cutoff = input$cutoff,
            date = Sys.Date()
          ),
          envir = new.env(parent = globalenv())
        )
      }
    )
}

shinyApp(ui, server)

