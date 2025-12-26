# ============================================================
# 🎯 RFE Ultimate App + HTML Report Export (Final Version)
# ============================================================

library(shiny)
library(caret)
library(mlbench)
library(ggplot2)
library(DT)
library(gridExtra)
library(dplyr)
library(randomForest)
library(tibble)
library(rmarkdown)

options(shiny.maxRequestSize = 50 * 1024^2)

# ------------------ UI ------------------
ui <- fluidPage(
  titlePanel("🎯 RFE Ultimate App"),
  
  sidebarLayout(
    sidebarPanel(
      fileInput("file", "Upload File (.tsv)", accept = ".tsv"),
      uiOutput("col_cat_ui"),
      uiOutput("modelo_ui"),
      selectInput("metrica", "Metric:", choices = c("Accuracy", "Kappa", "ROC", "RMSE", "Rsquared")),
      numericInput("num_vars", "Number of Variables to Download:", value = 30, min = 1, step = 1),
      numericInput("num_vars_plot", "Number of Variables to Show in Importance Plot:", value = 30, min = 5, step = 5),
      actionButton("run", "Run RFE"),
      hr(),
      downloadButton("downloadVars", "⬇️ Download Top Variables"),
      downloadButton("downloadReduced", "⬇️ Download Reduced Dataset"),
      downloadButton("downloadPlot", "⬇️ Download Curves PDF"),
      downloadButton("downloadImpPDF", "⬇️ Download Importance PDF"),
      downloadButton("downloadImpTSV", "⬇️ Download Importance TSV"),
      downloadButton("downloadHTMLReport", "📄 Download HTML Report") # 🆕
    ),
    
    mainPanel(
      tabsetPanel(
        tabPanel("Results", tableOutput("tablaResultados")),
        tabPanel("Curves", plotOutput("plotAccuracyKappa")),
        tabPanel("Importance", 
                 plotOutput("plotVarImp"),
                 dataTableOutput("tablaVarImp")),
        tabPanel("Top Variables", dataTableOutput("tablaVars"))
      )
    )
  )
)

# ------------------ SERVER ------------------
server <- function(input, output, session) {
  
  # === Load Data ===
  datos <- reactive({
    req(input$file)
    read.delim(input$file$datapath, sep="\t", header=TRUE, row.names=1)
  })
  
  # === Dynamic Selectors ===
  output$col_cat_ui <- renderUI({
    req(datos())
    selectInput("col_cat", "Category / Response Column:", choices = colnames(datos()))
  })
  
  output$modelo_ui <- renderUI({
    req(input$col_cat)
    df <- datos()
    var_resp <- df[[input$col_cat]]
    
    if (is.numeric(var_resp)) {
      modelos <- c("Linear Regression" = "lm",
                   "Random Forest" = "rf",
                   "Treebag" = "treebag",
                   "XGBoost" = "xgbTree")
    } else {
      modelos <- c("Random Forest" = "rf",
                   "Treebag" = "treebag",
                   "Naive Bayes" = "nb",
                   "CART" = "rpart",
                   "SVM Radial" = "svmRadial",
                   "XGBoost" = "xgbTree")
    }
    selectInput("modelo", "RFE Model:", choices = modelos)
  })
  
  # === RFE Training ===
  resultados <- eventReactive(input$run, {
    df <- datos()
    req(input$col_cat)
    
    y <- df[[input$col_cat]]
    x <- df[, setdiff(names(df), input$col_cat), drop = FALSE]
    
    process <- preProcess(x, method = c("range"))
    x <- predict(process, x)
    
    p <- ncol(x)
    subsets <- unique(c(1:5, seq(10, min(30, p), by = 5)))
    
    if (is.numeric(y)) {
      trctrl <- trainControl(method = "cv", number = 5)
    } else {
      y <- as.factor(y)
      trctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE, 
                             summaryFunction = twoClassSummary, allowParallel = TRUE)
    }
    
    if (input$modelo == "rf") {
      funcs <- rfFuncs; metodo_rfe <- NULL
    } else if (input$modelo == "treebag") {
      funcs <- treebagFuncs; metodo_rfe <- NULL
    } else if (input$modelo == "nb") {
      funcs <- nbFuncs; metodo_rfe <- NULL
    } else if (input$modelo == "lm") {
      funcs <- lmFuncs; metodo_rfe <- NULL
    } else {
      funcs <- caretFuncs; metodo_rfe <- input$modelo
    }
    
    rfe_ctrl <- rfeControl(functions = funcs, method = "cv", number = 5)
    
    set.seed(123)
    results <- rfe(
      x, y,
      sizes = subsets,
      rfeControl = rfe_ctrl,
      metric = input$metrica,
      method = metodo_rfe,
      trControl = trctrl
    )
    
    top_vars <- predictors(results)
    df_rf <- df[, c(input$col_cat, top_vars), drop = FALSE]
    y_rf <- df_rf[[input$col_cat]]
    x_rf <- df_rf[, setdiff(names(df_rf), input$col_cat), drop = FALSE]
    
    y_rf <- if (is.numeric(y_rf)) y_rf else as.factor(y_rf)
    rf_model <- randomForest(x = x_rf, y = y_rf, importance = TRUE)
    imp <- importance(rf_model)
    
    imp_col_index <- ifelse(is.numeric(y_rf), which(colnames(imp) == "%IncMSE"), 
                            which(colnames(imp) == "MeanDecreaseAccuracy"))
    
    imp_df <- data.frame(feature = rownames(imp),
                         importance = imp[, imp_col_index], 
                         row.names = NULL) %>%
      arrange(desc(importance))
    
    list(results = results, y = y, df = df, imp_df = imp_df, modelo = input$modelo, metrica = input$metrica)
  })
  
  # === Outputs ===
  output$tablaResultados <- renderTable({
    req(resultados())
    as.data.frame(resultados()$results$results)
  })
  
  output$plotAccuracyKappa <- renderPlot({
    req(resultados())
    print(plot(resultados()$results, type = c("g", "o")))
  })
  
  output$plotVarImp <- renderPlot({
    req(resultados())
    imp_df <- resultados()$imp_df
    top_imp <- head(imp_df, input$num_vars_plot)
    
    ggplot(top_imp, aes(x=reorder(feature, importance), y=importance, fill=feature)) +
      geom_bar(stat="identity") + coord_flip() + theme_bw() +
      labs(x="Variables", y="Importance (Auxiliary RF)") + guides(fill="none")
  })
  
  output$tablaVarImp <- renderDataTable({
    req(resultados())
    datatable(resultados()$imp_df, options = list(pageLength = 15))
  })
  
  output$tablaVars <- renderDataTable({
    req(resultados())
    data.frame(Top_Variables = predictors(resultados()$results))
  })
  
  # === Downloads ===
  output$downloadVars <- downloadHandler(
    filename = function() { "Top_Variables.tsv" },
    content = function(file) {
      vars <- data.frame(Top_Variables = predictors(resultados()$results))
      write.table(vars, file, sep="\t", quote=FALSE, row.names=FALSE)
    }
  )
  
  output$downloadReduced <- downloadHandler(
    filename = function() { "Reduced_Dataset.tsv" },
    content = function(file) {
      res <- resultados()
      top_vars <- head(res$imp_df$feature, input$num_vars)
      df_reducido <- res$df[, c(input$col_cat, top_vars), drop=FALSE]
      df_reducido <- tibble::rownames_to_column(df_reducido, var = "SampleID")
      write.table(df_reducido, file, sep="\t", quote=FALSE, row.names=FALSE)
    }
  )
  
  output$downloadPlot <- downloadHandler(
    filename = function() { "RFE_Curves.pdf" },
    content = function(file) {
      pdf(file, width=12, height=8)
      print(plot(resultados()$results, type=c("g","o")))
      dev.off()
    }
  )
  
  output$downloadImpPDF <- downloadHandler(
    filename = function() { "Variable_Importance_RF.pdf" },
    content = function(file) {
      imp_df <- resultados()$imp_df
      top_imp <- head(imp_df, input$num_vars_plot)
      pdf(file, width=10, height=6)
      print(ggplot(top_imp, aes(x=reorder(feature, importance), y=importance, fill=feature)) +
              geom_bar(stat="identity") + coord_flip() + theme_bw() +
              labs(x="Variables", y="Importance (Auxiliary RF)") + guides(fill="none"))
      dev.off()
    }
  )
  
  output$downloadImpTSV <- downloadHandler(
    filename = function() { "Variable_Importance_RF.tsv" },
    content = function(file) {
      write.table(resultados()$imp_df, file, sep="\t", quote=FALSE, row.names=FALSE)
    }
  )
  
  # === 🆕 HTML Report (RMarkdown) ===
  output$downloadHTMLReport <- downloadHandler(
    filename = function() { paste0("RFE_Report_", Sys.Date(), ".html") },
    content = function(file) {
      temp_rmd <- tempfile(fileext = ".Rmd")
      rmd_lines <- c(
        "---",
        "title: 'RFE Ultimate Analysis Report'",
        "output: html_document",
        "params:",
        "  modelo: NULL",
        "  metrica: NULL",
        "  results: NULL",
        "  imp_df: NULL",
        "  top_vars: NULL",
        "  date: NULL",
        "---",
        "",
        "```{r setup, include=FALSE}",
        "library(ggplot2); library(dplyr); library(gridExtra); library(caret);",
        "knitr::opts_chunk$set(echo=FALSE, warning=FALSE, message=FALSE)",
        "```",
        "",
        "# 🎯 RFE Ultimate Analysis Report",
        "",
        "**Date:** `r params$date`  ",
        "",
        "**Model Used:** `r params$modelo`  ",
        "",
        "**Metric Optimized:** `r params$metrica`  ",
        "",
        "## 🔹 RFE Performance Curves",
        "```{r}",
        "plot(params$results, type=c('g','o'))",
        "```",
        "",
        "## 🔹 Variable Importance (Auxiliary Random Forest)",
        "```{r}",
        "top_imp <- head(params$imp_df, 30)",
        "ggplot(top_imp, aes(x=reorder(feature, importance), y=importance, fill=feature)) +",
        " geom_bar(stat='identity') + coord_flip() + theme_bw() + guides(fill='none') +",
        " labs(x='Variables', y='Importance')",
        "```",
        "",
        "## 🔹 Top Variables Selected by RFE",
        "```{r}",
        "data.frame(Top_Variables = params$top_vars)",
        "```",
        "",
        "## 🔹 Variable Importance Table (First 20)",
        "```{r}",
        "head(params$imp_df, 20)",
        "```"
      )
      writeLines(rmd_lines, temp_rmd)
      
      res <- resultados()
      rmarkdown::render(
        input = temp_rmd,
        output_file = file,
        params = list(
          modelo = res$modelo,
          metrica = res$metrica,
          results = res$results,
          imp_df = res$imp_df,
          top_vars = predictors(res$results),
          date = Sys.Date()
        ),
        envir = new.env(parent = globalenv())
      )
    }
  )
}

# ------------------ Run App ------------------
shinyApp(ui, server)
