# app.R - SHAP Batch Analysis App (self-contained, dynamic RMarkdown report)
library(shiny)
library(caret)
library(randomForest)
library(fastshap)
library(ggplot2)
library(dplyr)
library(tidyr)
library(future)
library(future.apply)
library(readr)
library(rmarkdown)

options(shiny.maxRequestSize = 50 * 1024^2)

# ==============================================================================
# 1. SHAP Core Functions
# ==============================================================================

compute_shap <- function(model, data, class_label, nsim, sample_size) {
  if (!is.null(sample_size) && sample_size < nrow(data)) {
    set.seed(42)
    data <- data[sample(nrow(data), sample_size), ]
  }
  
  pred_fun <- function(object, newdata) {
    predict(object, newdata = newdata, type = "prob")[, class_label]
  }
  
  target_col <- names(data)[sapply(data, is.factor)][1]
  X_data <- data %>% select(-all_of(target_col))
  
  shap_values <- fastshap::explain(
    object = model,
    X = X_data,
    pred_wrapper = pred_fun,
    nsim = nsim,
    adjust = TRUE
  )
  
  shap_long <- as.data.frame(shap_values) %>%
    mutate(row = row_number()) %>%
    pivot_longer(-row, names_to = "variable", values_to = "shap") %>%
    group_by(variable) %>%
    summarise(mean_abs_shap = mean(abs(shap), na.rm = TRUE), .groups = "drop")
  
  shap_plot <- ggplot(shap_long, aes(x = reorder(variable, mean_abs_shap), y = mean_abs_shap)) +
    geom_col(fill = "#2C77B8") +
    coord_flip() +
    labs(
      title = paste("SHAP Feature Importance - Class:", class_label),
      x = "Feature",
      y = "Mean Absolute SHAP Value"
    ) +
    theme_minimal(base_size = 14) +
    theme(plot.title = element_text(hjust = 0.5))
  
  return(list(summary = shap_long, plot = shap_plot, class = class_label))
}

# ==============================================================================
# 2. Shiny UI
# ==============================================================================

ui <- fluidPage(
  titlePanel("✨ Multiclass SHAP Batch Analysis"),
  
  sidebarLayout(
    sidebarPanel(
      fileInput("file", "1. Upload Data File (.tsv)", accept = ".tsv"),
      uiOutput("target_col_ui"),
      hr(),
      h4("SHAP & Model Settings"),
      numericInput("nsim", "SHAP Simulations (nsim):", value = 100, min = 50, step = 10),
      numericInput("sample_size", "Sample Size for SHAP (max 500):", value = 200, min = 50, max = 500, step = 50),
      actionButton("run_analysis", "3. Train Model & Run SHAP Batch", icon = icon("rocket")),
      hr(),
      h4("Download Results"),
      downloadButton("downloadSummary", "⬇️ Download Combined SHAP CSV"),
      downloadButton("downloadPlots", "⬇️ Download All SHAP Plots (PDF)"),
      hr(),
      h4("Generate Report (HTML Only)"),
      downloadButton("downloadReport", "📄 Download SHAP HTML Report")
    ),
    
    mainPanel(
      tabsetPanel(
        tabPanel("Input Data", dataTableOutput("input_data_table")),
        tabPanel("Model Status", verbatimTextOutput("model_summary")),
        tabPanel("SHAP Plots (Batch Results)", uiOutput("shap_plots_ui"))
      )
    )
  )
)

# ==============================================================================
# 3. Shiny Server
# ==============================================================================

server <- function(input, output, session) {
  
  # --- Data Loading ---
  data_df <- reactive({
    req(input$file)
    df <- read_tsv(input$file$datapath, show_col_types = FALSE)
    if (any(names(df) %in% c("X", "SampleID"))) {
      df <- df %>% select(-all_of(c("X", "SampleID")[c("X", "SampleID") %in% names(df)]))
    }
    df
  })
  
  output$target_col_ui <- renderUI({
    df <- data_df(); req(df)
    selectInput("target_col", "2. Select Target/Response Column:", 
                choices = names(df), selected = names(df)[1])
  })
  
  output$input_data_table <- renderDataTable({
    data_df()
  })
  
  # --- Model Training & SHAP Calculation ---
  analysis_results <- eventReactive(input$run_analysis, {
    df <- data_df(); req(df, input$target_col)
    
    local_nsim <- input$nsim
    local_sample_size <- input$sample_size
    
    df_clean <- df
    target_name <- input$target_col
    df_clean[[target_name]] <- factor(make.names(df_clean[[target_name]]))
    
    y <- df_clean[[target_name]]
    X <- df_clean %>% select(-all_of(target_name))
    
    withProgress(message = "Training Random Forest...", value = 0.3, {
      set.seed(42)
      model_fit <- train(x = X, y = y,
                         method = "rf",
                         trControl = trainControl(method = "none"),
                         tuneGrid = data.frame(mtry = sqrt(ncol(X))))
    })
    
    classes <- levels(y)
    withProgress(message = "Calculating SHAP values for all classes...", value = 0.6, {
      plan(multisession, workers = min(length(classes), availableCores()))
      results <- future_lapply(classes, function(cl) {
        compute_shap(model_fit, df_clean, cl, local_nsim, local_sample_size)
      }, future.seed = TRUE)
      plan(sequential)
    })
    
    combined_summary <- bind_rows(lapply(results, function(r) {
      mutate(r$summary, class = r$class)
    }))
    
    list(model = model_fit, results = results, combined_summary = combined_summary)
  })
  
  # --- Outputs ---
  output$model_summary <- renderPrint({
    req(analysis_results())
    print(analysis_results()$model)
  })
  
  output$shap_plots_ui <- renderUI({
    res <- analysis_results(); req(res)
    tagList(lapply(res$results, function(r) {
      tagList(h4(paste("Class:", r$class)), renderPlot({ r$plot }, width = 700, height = 400))
    }))
  })
  
  # --- Downloads ---
  output$downloadSummary <- downloadHandler(
    filename = function() { "SHAP_Combined_Summary.csv" },
    content = function(file) {
      write_csv(analysis_results()$combined_summary, file)
    }
  )
  
  output$downloadPlots <- downloadHandler(
    filename = function() { "SHAP_All_Class_Plots.pdf" },
    content = function(file) {
      res <- analysis_results(); req(res)
      pdf(file, width = 10, height = 7)
      lapply(res$results, function(r) print(r$plot))
      dev.off()
    }
  )
  
  # --- Report Generation (HTML Only) ---
  output$downloadReport <- downloadHandler(
    filename = function() {
      paste0("SHAP_Report_", Sys.Date(), ".html")
    },
    content = function(file) {
      res <- analysis_results(); req(res)
      
      # Crear Rmd temporal
      temp_rmd <- file.path(tempdir(), "report.Rmd")
      rmd_lines <- c(
        "---",
        "title: 'SHAP Batch Analysis Report'",
        "output: html_document",
        "params:",
        "  res: NULL",
        "  date: NA",
        "---",
        "",
        "```{r setup, include=FALSE}",
        "library(ggplot2)",
        "library(dplyr)",
        "knitr::opts_chunk$set(echo = FALSE, warning = FALSE, message = FALSE)",
        "```",
        "",
        "# 🧠 SHAP Batch Analysis Report",
        "",
        "**Fecha del análisis:** `r params$date`  ",
        "",
        "## 📈 Modelo Entrenado",
        "```{r}",
        "print(params$res$model)",
        "```",
        "",
        "## 📊 Resumen Combinado SHAP",
        "```{r}",
        "head(params$res$combined_summary, 20)",
        "```",
        "",
        "## 🔍 Importancia de Características por Clase",
        "```{r, results='asis'}",
        "for (r in params$res$results) {",
        "  cat('### Clase:', r$class, '\\n')",
        "  print(r$plot)",
        "  cat('\\n\\n')",
        "}",
        "```"
      )
      
      writeLines(rmd_lines, temp_rmd)
      
      # Render HTML
      rmarkdown::render(
        input = temp_rmd,
        output_file = file,
        params = list(res = res, date = Sys.Date()),
        envir = new.env(parent = globalenv())
      )
    }
  )
}

# --- Launch App ---
shinyApp(ui, server)


