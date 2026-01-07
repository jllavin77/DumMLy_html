library(shiny)
library(readr)
library(ggplot2)
library(dplyr)
library(magrittr)
library(caret)
library(automl)
library(rmarkdown)

options(shiny.maxRequestSize = 50 * 1024^2) # Max 50 MB upload

ui <- fluidPage(
  titlePanel("Automated Machine Learning (AutoML) with R"),
  sidebarLayout(
    sidebarPanel(
      fileInput("file", "1. Upload TSV Data", accept = ".tsv"),
      uiOutput("target_ui"),
      hr(),
      h4("AutoML Parameters (Automatic Tunning)"),
      numericInput("iterations", "Number of iterations (numiterations):", value = 5, min = 1, max = 50),
      numericInput("popsize", "PSO Population Size (psopartpopsize):", value = 15, min = 5, max = 50),
      numericInput("layers_max", "Max Layers (auto_layers_max):", value = 1, min = 1, max = 5),
      hr(),
      actionButton("train_automl", "2. Train AutoML Model", class = "btn-primary"),
      hr(),
      downloadButton("download_model", "⬇️ Download Model (RDS)"),
      downloadButton("download_html_report", "⬇️ Download HTML Report")
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("Input Data Head",
                 h4("First 10 Rows of Input Data"),
                 tableOutput("data_head_preview")
        ),
        tabPanel("Metrics & Summary",
                 h4("Model Training Metrics"),
                 tableOutput("metrics_table"),
                 h4("Training Time Elapsed"),
                 textOutput("time_elapsed")
        ),
        tabPanel("Predictions (Head)",
                 h4("Predictions vs. Actual (Training Data)"),
                 tableOutput("predictions_head")
        ),
        tabPanel("Variable Importance",
                 h4("Variable Importance (Absolute Correlation with Target)"),
                 p("Note: The 'automl' package does not provide standard importance metrics. This table displays the absolute correlation between each predictor and the target variable as a proxy for importance."),
                 tableOutput("varimp_table"),
                 plotOutput("varimp_plot")
        )
      )
    )
  )
)

server <- function(input, output, session) {
  
  automl_results <- reactiveVal(NULL)
  
  # === 1. Data Loading and Cleaning ===
  
  data_input <- reactive({
    req(input$file)
    df <- read_tsv(input$file$datapath, show_col_types = FALSE)
    
    if ("X" %in% names(df)) { df$X <- NULL }
    colnames(df) <- gsub("X.", "", colnames(df), fixed = TRUE)
    
    return(df)
  })
  
  output$data_head_preview <- renderTable({
    req(data_input())
    head(data_input(), 10)
  })
  
  output$target_ui <- renderUI({
    req(data_input())
    df_names <- names(data_input())
    default_target <- if ("Carcass_Category_0" %in% df_names) "Carcass_Category_0" else df_names[1]
    
    selectInput("target_col", "Select target variable:",
                choices = df_names,
                selected = default_target)
  })
  
  variable_importance_data <- reactive({
    req(data_input(), input$target_col)
    df <- data_input()
    target <- input$target_col
    
    df_numeric <- df %>% select(where(is.numeric))
    
    if (!target %in% names(df_numeric)) {
      if (is.character(df[[target]]) || is.factor(df[[target]])) {
        target_numeric <- as.numeric(as.factor(df[[target]]))
      } else {
        return(data.frame(Variable = "N/A", Importance = 0)[0, ])
      }
      df_predictors <- df_numeric 
    } else {
      target_numeric <- df_numeric[[target]]
      df_predictors <- df_numeric %>% select(-all_of(target))
    }
    
    if (ncol(df_predictors) == 0) {
      return(data.frame(Variable = "N/A", Importance = 0)[0, ])
    }
    
    cor_values <- cor(df_predictors, target_numeric, use = "pairwise.complete.obs")
    
    data.frame(
      Variable = rownames(cor_values),
      Importance = abs(cor_values[, 1])
    ) %>%
      arrange(desc(Importance))
  })
  
  # === 2. AutoML Training Logic (Event Reactive) ===
  
  training_results <- eventReactive(input$train_automl, {
    req(data_input(), input$target_col)
    
    df <- data_input()
    target <- input$target_col
    
    # Split data
    set.seed(42)
    train_indices <- sample(1:nrow(df), 0.7 * nrow(df))
    train_data <- df[train_indices, ]
    
    # Define Yref 
    ymat_raw <- train_data[[target]]
    
    if (is.character(ymat_raw) || is.factor(ymat_raw) || length(unique(ymat_raw)) < 10) {
      task_type <- "Classification (Ordinal)"
      ymat <- as.numeric(as.factor(ymat_raw))
    } else {
      task_type <- "Regression"
      ymat <- as.numeric(ymat_raw)
    }
    
    # Filtrar solo columnas numéricas para Xref
    xmat <- train_data %>%
      select(-all_of(target)) %>%
      select(where(is.numeric))
    
    if (ncol(xmat) == 0) {
      showNotification("⚠️ Error: No hay variables predictoras numéricas disponibles después del filtrado.", type = "error", duration = 8)
      automl_results(NULL)
      return(NULL)
    }
    
    # --- Automated Tunning ---
    showNotification("Starting AutoML Training...", duration = NULL, type = "message")
    
    # 🚨 CORRECCIÓN: Usar base::Sys.time()
    start_time <- base::Sys.time()
    
    amlmodel <- tryCatch({
      automl_train(
        Xref = xmat,
        Yref = ymat,
        autopar = list(
          psopartpopsize = input$popsize,
          numiterations = input$iterations,
          auto_layers_max = input$layers_max,
          nbcores = 1
        )
      )
    }, error = function(e) {
      showNotification(paste("AutoML Training Failed:", e$message), type = "error", duration = 8)
      return(NULL)
    })
    
    if (is.null(amlmodel)) {
      automl_results(NULL)
      return(NULL)
    }
    
    # 🚨 CORRECCIÓN: Usar base::Sys.time()
    end_time <- base::Sys.time()
    time_elapsed <- end_time - start_time
    
    # --- Prediction and Metrics ---
    
    prediction_train_raw <- automl_predict(model = amlmodel, X = xmat)
    
    metrics <- NULL
    predictions_df <- data.frame(Actual = ymat_raw)
    
    if (task_type == "Classification (Ordinal)") {
      pred_train_rounded <- prediction_train_raw %>% round() %>% as.factor()
      
      cm <- tryCatch(
        confusionMatrix(pred_train_rounded, as.factor(ymat)),
        error = function(e) { NULL }
      )
      
      if (!is.null(cm)) {
        metrics <- data.frame(
          Task = task_type,
          Metric = cm$overall["Accuracy"],
          Metric_Name = "Accuracy",
          Prediction_Basis = "Rounded Training Data"
        )
      } else {
        rmse_val <- RMSE(prediction_train_raw, ymat)
        metrics <- data.frame(
          Task = task_type,
          Metric = rmse_val,
          Metric_Name = "RMSE (Raw Prediction)",
          Prediction_Basis = "Raw Training Data"
        )
      }
      
      predictions_df$Predicted <- pred_train_rounded
      
    } else { # Regression
      rmse_val <- RMSE(prediction_train_raw, ymat)
      metrics <- data.frame(
        Task = task_type,
        Metric = rmse_val,
        Metric_Name = "RMSE",
        Prediction_Basis = "Training Data"
      )
      predictions_df$Predicted <- prediction_train_raw
    }
    
    showNotification("AutoML Training Complete!", type = "default", duration = 3)
    
    result_list <- list(
      model = amlmodel,
      metrics = metrics,
      predictions = predictions_df,
      time = time_elapsed,
      task = task_type,
      target = target
    )
    
    automl_results(result_list)
    return(result_list)
  })
  
  # === 3. Outputs ===
  
  output$metrics_table <- renderTable({ req(training_results()); training_results()$metrics })
  # 🚨 CORRECCIÓN: Llamar Sys.time en el servidor no debería tener problema, pero la métrica ya está en el objeto results.
  output$time_elapsed <- renderText({ req(training_results()); paste("Time elapsed:", round(training_results()$time, 2), attr(training_results()$time, "units")) })
  output$predictions_head <- renderTable({ req(training_results()); head(training_results()$predictions, 10) })
  
  output$varimp_table <- renderTable({
    req(variable_importance_data())
    df_imp <- variable_importance_data()
    if (nrow(df_imp) == 0) {
      return(data.frame(Message = "Cannot compute correlation importance (No numeric predictors)."))
    }
    return(df_imp)
  })
  
  output$varimp_plot <- renderPlot({
    req(variable_importance_data())
    df_imp <- variable_importance_data()
    if (nrow(df_imp) == 0) return(NULL)
    
    ggplot(df_imp %>% top_n(20, Importance), 
           aes(x = reorder(Variable, Importance), y = Importance)) +
      geom_col(fill = "#007bff") +
      coord_flip() +
      theme_minimal(base_size = 14) +
      labs(title = "Top 20 Variable Importance (Absolute Correlation)",
           y = "Absolute Correlation Value",
           x = "Variable")
  })
  
  # === 4. Downloads ===
  
  output$download_model <- downloadHandler(
    filename = function() { "automl_model.rds" },
    content = function(file) { req(training_results()); saveRDS(training_results()$model, file = file) }
  )
  
  # === 5. Dynamic HTML Report (R Markdown) ===
  output$download_html_report <- downloadHandler(
    # 🚨 CORRECCIÓN: Usar base::Sys.Date()
    filename = function() paste0("AutoML_Report_", base::Sys.Date(), ".html"),
    content = function(file) {
      
      req(training_results())
      current_results <- isolate(training_results())
      varimp_data <- isolate(variable_importance_data())
      
      params_list <- list(
        metrics = current_results$metrics,
        predictions_head = head(current_results$predictions, 20),
        varimp = varimp_data,
        time = current_results$time,
        task = current_results$task,
        target = current_results$target,
        # 🚨 CORRECCIÓN: Usar base::Sys.Date()
        date = base::Sys.Date()
      )
      
      temp_rmd <- tempfile(fileext = ".Rmd")
      rmd_lines <- c(
        "---", "title: 'AutoML Training Report'", "output: html_document", "params:",
        "  metrics: NULL", "  predictions_head: NULL", "  varimp: NULL", "  time: NULL", "  task: NULL", "  target: NULL", "  date: NULL",
        "---",
        "",
        "```{r setup, include=FALSE}", "library(knitr); library(dplyr); library(ggplot2);", "knitr::opts_chunk$set(echo=FALSE, warning=FALSE, message=FALSE, fig.width=9, fig.height=6)",
        "```",
        "",
        "# 🤖 AutoML Training Report", "", "**Target Variable:** `r params$target`", "**Inferred Task Type:** `r params$task`", "**Report Date:** `r params$date`",
        "",
        "## ⏱️ Training Summary", "", "**Time Elapsed for Training:** `r round(params$time, 2)` `r attr(params$time, \"units\")`",
        "",
        "## 📊 Model Performance Metrics", "The primary metric achieved on the **training data** (70% split) is shown below:",
        "```{r}", "print(params$metrics)", "```",
        "",
        "## 📈 Variable Importance (Correlation Proxy)", "This metric is calculated as the absolute correlation between each numeric predictor and the target variable.",
        "### Table",
        "```{r}", "if (nrow(params$varimp) > 0) {", "  print(params$varimp)", "} else {", "  cat('Variable importance data is not available (No numeric predictors or target issues).')", "}", "```",
        "### Plot",
        "```{r}", "if (nrow(params$varimp) > 0) {", "  df_imp <- params$varimp %>% top_n(20, Importance)", "  ggplot(df_imp, aes(x = reorder(Variable, Importance), y = Importance)) +", "    geom_col(fill = '#007bff') +", "    coord_flip() +", "    theme_minimal(base_size = 14) +", "    labs(title = 'Top 20 Variable Importance (Absolute Correlation)', y = 'Absolute Correlation Value', x = 'Variable')", "} else {", "  # Placeholder for empty plot", "}", "```",
        "",
        "## 🔍 Prediction Comparison (First 20 Samples)", "Comparison of Actual vs. Predicted values from the trained model on the training subset:",
        "```{r}", "print(params$predictions_head)", "```",
        "",
        "---", "This report was automatically generated using the `automl` R package and Shiny."
      )
      writeLines(rmd_lines, temp_rmd)
      
      rmarkdown::render(
        input = temp_rmd,
        output_file = file,
        params = params_list,
        envir = new.env(parent = globalenv())
      )
    }
  )
  
}

shinyApp(ui = ui, server = server)
