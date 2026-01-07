library(shiny)
library(readr)
library(caret)
library(neuralnet)
library(ggplot2)
library(dplyr)
library(rmarkdown) # Aseguramos que rmarkdown esté cargado

# library(keras3) # Se mantiene comentado
# library(tensorflow) # Se mantiene comentado

options(shiny.maxRequestSize = 50 * 1024^2) # Up to 50 MB

ui <- fluidPage(
  titlePanel("Neural Network Comparator"),
  sidebarLayout(
    sidebarPanel(
      fileInput("file", "Upload your TSV File", accept = ".tsv"),
      uiOutput("target_ui"),
      selectInput("modelo", "Model to train:",
                  choices = c("All", "nnet (caret)", "neuralnet (caret)", "keras"),
                  selected = "All"),
      actionButton("train", "Train Models"),
      hr(),
      downloadButton("download_results", "Download Metrics (CSV)"),
      downloadButton("download_models", "Download Models (RDS)"),
      downloadButton("download_html_report", "⬇️ Download HTML Report") # Botón Añadido
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("Summary",
                 tableOutput("results")),
        tabPanel("Variable Importance",
                 plotOutput("varimp_plot"),
                 tableOutput("varimp_table"))
      )
    )
  )
)

server <- function(input, output, session) {
  
  models_list <- reactiveVal(list()) # Reactive value para guardar los modelos
  
  # Read data
  data <- reactive({
    req(input$file)
    read_tsv(input$file$datapath, show_col_types = FALSE)
  })
  
  # Dynamic UI for target variable
  output$target_ui <- renderUI({
    req(data())
    selectInput("target", "Select target variable:",
                choices = names(data()))
  })
  
  # Train models
  results <- eventReactive(input$train, {
    req(data(), input$target)
    
    df <- data()
    
    # Remove columns with all unique values (e.g., IDs)
    df <- df[, sapply(df, function(col) length(unique(col)) < nrow(df))]
    
    # Define target and features
    target <- input$target
    
    # Detect problem type and preprocess target
    if (is.numeric(df[[target]])) {
      task <- "regression"
      metric <- "RMSE"
    } else {
      task <- "classification"
      metric <- "Accuracy"
      df[[target]] <- factor(make.names(df[[target]])) # Ensure target is a factor with valid names
      if (length(levels(df[[target]])) < 2) {
        stop("Classification target must have at least 2 unique classes.")
      }
    }
    
    # Filter predictors with near zero variance
    nzv <- nearZeroVar(df, saveMetrics = TRUE, names = TRUE)
    nzv_cols <- rownames(nzv[nzv$nzv,])
    df <- df |> select(-all_of(nzv_cols))
    
    # Final predictor list
    predictors <- setdiff(names(df), target)
    if (length(predictors) == 0) {
      stop("No suitable predictor columns remaining after filtering.")
    }
    
    # Data partitioning (must be done AFTER target cleanup)
    trainIndex <- createDataPartition(df[[target]], p = .7, list = FALSE)
    train <- df[trainIndex, ]
    test <- df[-trainIndex, ]
    
    # Control for caret models (CV and preProcess)
    fitControl <- trainControl(
      method = "cv", 
      number = 3, 
      savePredictions = "final",
      classProbs = task == "classification", # Required for multi-class nnet
      summaryFunction = if (task == "classification") defaultSummary else defaultSummary
    )
    
    modelos <- list()
    metricas <- list()
    varimps <- list()
    
    # --- nnet (caret) ---
    if (input$modelo %in% c("All", "nnet (caret)")) {
      tuneGrid <- expand.grid(size = c(3, 5), decay = c(0.001))
      
      model_nnet <- tryCatch({
        withProgress(message = 'Training nnet...', value = 0.3, {
          train(as.formula(paste(target, "~ .")),
                data = train,
                method = "nnet",
                trControl = fitControl,
                tuneGrid = tuneGrid,
                preProcess = c("center", "scale"), # ESSENTIAL PREPROCESSING
                trace = FALSE,
                metric = metric)
        })
      }, error = function(e) {
        showNotification(paste("⚠️ nnet failed:", e$message), type = "error")
        NULL
      })
      
      if (!is.null(model_nnet)) {
        modelos[["nnet"]] <- model_nnet
        pred <- predict(model_nnet, newdata = test)
        
        if (task == "classification") {
          cm <- confusionMatrix(pred, test[[target]])
          metricas[["nnet"]] <- data.frame(Model = "nnet", Accuracy = cm$overall["Accuracy"])
        } else {
          rmse <- RMSE(pred, test[[target]])
          metricas[["nnet"]] <- data.frame(Model = "nnet", RMSE = rmse)
        }
        
        # Variable Importance
        vi <- varImp(model_nnet)$importance
        vi$Variable <- rownames(vi)
        vi$Model <- "nnet"
        varimps[["nnet"]] <- vi
      }
    }
    
    # --- neuralnet (caret) ---
    if (input$modelo %in% c("All", "neuralnet (caret)")) {
      model_nn <- tryCatch({
        withProgress(message = 'Training neuralnet...', value = 0.6, {
          train(as.formula(paste(target, "~ .")),
                data = train,
                method = "neuralnet",
                trControl = fitControl,
                preProcess = c("center", "scale", "spatialSign"), # Stronger preprocessing
                metric = metric)
        })
      }, error = function(e) {
        showNotification(paste("⚠️ neuralnet failed:", e$message), type = "error")
        NULL
      })
      
      if (!is.null(model_nn)) {
        modelos[["neuralnet"]] <- model_nn
        pred <- predict(model_nn, newdata = test)
        
        if (task == "classification") {
          cm <- confusionMatrix(pred, test[[target]])
          metricas[["neuralnet"]] <- data.frame(Model = "neuralnet", Accuracy = cm$overall["Accuracy"])
        } else {
          rmse <- RMSE(pred, test[[target]])
          metricas[["neuralnet"]] <- data.frame(Model = "neuralnet", RMSE = rmse)
        }
        
        # Variable Importance (handle potential failure)
        vi <- tryCatch({
          vi_data <- varImp(model_nn)$importance
          vi_data$Variable <- rownames(vi_data)
          vi_data$Model <- "neuralnet"
          vi_data
        }, error = function(e) {
          # Retorna un data frame vacío si falla
          data.frame(Overall = numeric(), Variable = character(), Model = character())
        })
        varimps[["neuralnet"]] <- vi
      }
    }
    
    # --- Keras/TensorFlow ---
    if (input$modelo %in% c("All", "keras")) {
      if (requireNamespace("keras3", quietly = TRUE) && requireNamespace("tensorflow", quietly = TRUE)) {
        # ... Keras training block ...
        # Placeholder for Keras logic. Assuming it adds results to 'modelos', 'metricas', 'varimps' on success.
      } else {
        showNotification("Keras/TensorFlow not installed or configured.", type = "warning")
      }
    }
    
    # Update reactive values
    models_list(modelos)
    
    # Ensure do.call doesn't break if lists are empty
    metricas_df <- if (length(metricas) > 0) do.call(rbind, metricas) else data.frame(Model=character(), Accuracy=numeric(), RMSE=numeric()) # Asegura columnas si está vacío
    varimps_df <- if (length(varimps) > 0) do.call(rbind, varimps) else data.frame(Overall = numeric(), Variable = character(), Model = character()) # Asegura columnas si está vacío
    
    # Return results
    list(metricas = metricas_df,
         varimps = varimps_df)
  })
  
  # --- Outputs ---
  
  # Mostrar métricas 
  output$results <- renderTable({
    req(results())
    metrics_df <- results()$metricas
    
    if (is.null(metrics_df) || nrow(metrics_df) == 0) {
      return(data.frame(Message = "No models trained successfully or no results obtained."))
    }
    
    return(metrics_df)
  })
  
  # Variable Importance (table)
  output$varimp_table <- renderTable({
    req(results())
    varimp_df <- results()$varimps
    
    if (is.null(varimp_df) || nrow(varimp_df) == 0) {
      return(data.frame(Message = "Variable Importance not available for successful models."))
    }
    
    varimp_df |> arrange(desc(Overall))
  })
  
  # Variable Importance (plot)
  output$varimp_plot <- renderPlot({
    req(results())
    varimp_df <- results()$varimps
    
    if (is.null(varimp_df) || nrow(varimp_df) == 0) return(NULL)
    
    ggplot(varimp_df, aes(x = reorder(Variable, Overall), y = Overall, fill = Model)) +
      geom_col(position = "dodge") +
      coord_flip() +
      theme_minimal(base_size = 14) +
      labs(title = "Variable Importance", y = "Importance", x = "Variable")
  })
  
  # --- Downloads ---
  
  # Download results
  output$download_results <- downloadHandler(
    filename = function() { "model_results.csv" },
    content = function(file) {
      write.csv(results()$metricas, file, row.names = FALSE)
    }
  )
  
  # Download models
  output$download_models <- downloadHandler(
    filename = function() { "trained_models.rds" },
    content = function(file) {
      modelos <- models_list()
      saveRDS(modelos, file = file)
    }
  )
  
  # === Reporte HTML R Markdown (CORRECCIÓN INTEGRADA) ===
  output$download_html_report <- downloadHandler(
    filename = function() paste0("NN_Comparator_Report_", Sys.Date(), ".html"),
    content = function(file) {
      
      # 1. Requiere que los resultados existan (que se haya pulsado 'Train Models')
      req(results()) 
      
      # 2. CORRECCIÓN CLAVE: Extraer los valores de las funciones reactivas usando isolate()
      current_results <- isolate(results())
      metrics_df <- current_results$metricas
      varimp_df <- current_results$varimps
      
      # 3. Definir la lista de parámetros
      params_list <- list(
        metrics = metrics_df,
        varimp = varimp_df,
        target_var = input$target,
        date = Sys.Date()
      )
      
      # 4. Generar la plantilla Rmd (on the fly)
      temp_rmd <- tempfile(fileext = ".Rmd")
      rmd_lines <- c(
        "---",
        "title: 'Neural Network Comparison Report'",
        "output: html_document",
        "params:",
        "  metrics: NULL",
        "  varimp: NULL",
        "  target_var: NULL",
        "  date: NULL",
        "---",
        "",
        "```{r setup, include=FALSE}",
        "library(ggplot2); library(dplyr); library(knitr);",
        "knitr::opts_chunk$set(echo=FALSE, warning=FALSE, message=FALSE, fig.width=9, fig.height=6)",
        "```",
        "",
        "# 📈 Neural Network Comparison Report",
        "",
        "**Target Variable:** `r params$target_var`",
        "**Report Date:** `r params$date`",
        "",
        "## 🔹 Model Performance Summary",
        "The following table summarizes the metrics (Accuracy or RMSE) for the successfully trained models:",
        "```{r}",
        "if (nrow(params$metrics) > 0) {",
        "  print(params$metrics)",
        "} else {",
        "  cat('No model metrics available.')",
        "}",
        "```",
        "",
        "## 🔹 Variable Importance",
        "This section shows the relative importance of predictor variables across the trained models.",
        "### Table",
        "```{r}",
        "if (nrow(params$varimp) > 0) {",
        "  print(params$varimp |> arrange(desc(Overall)))",
        "} else {",
        "  cat('Variable Importance data is not available.')",
        "}",
        "```",
        "### Plot",
        "```{r}",
        "if (nrow(params$varimp) > 0) {",
        "  ggplot(params$varimp, aes(x = reorder(Variable, Overall), y = Overall, fill = Model)) +",
        "    geom_col(position = 'dodge') +",
        "    coord_flip() +",
        "    theme_minimal(base_size = 14) +",
        "    labs(title = 'Variable Importance', y = 'Importance', x = 'Variable')",
        "} else {",
        "  cat('No plot generated because Variable Importance data is not available.')",
        "}",
        "```"
      )
      writeLines(rmd_lines, temp_rmd)
      
      # 5. Renderizar el reporte
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
