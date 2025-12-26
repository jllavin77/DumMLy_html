library(shiny)
library(tidyverse)
library(ggplot2)
library(rmarkdown)
library(caret)

ui <- fluidPage(
  titlePanel("Multiple Model Prediction"),
  sidebarLayout(
    sidebarPanel(
      fileInput("datafile", "Upload your Data (CSV/TSV)", accept = c(".csv", ".tsv")),
      fileInput("modelfiles", "Upload one or more Models (.rds)", accept = ".rds", multiple = TRUE),
      uiOutput("modelo_sel_ui"),
      hr(),
      uiOutput("id_col_ui"),
      uiOutput("label_col_ui"),
      hr(),
      actionButton("go", "Predict with Selected Model"),
      actionButton("go_all", "Predict with ALL Models"),
      hr(),
      downloadButton("download_preds", "⬇️ Selected Predictions"),
      downloadButton("download_preds_all", "⬇️ Comparative Predictions"),
      downloadButton("download_heatmap", "⬇️ Download Heatmap (PDF)"),
      downloadButton("download_html_report", "⬇️ Download HTML Report")
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("Data and Predictions",
                 h4("Input Data Preview"),
                 tableOutput("preview"),
                 h4("Individual Predictions"),
                 tableOutput("preds"),
                 h4("Comparative Predictions"),
                 tableOutput("preds_all")
        ),
        tabPanel("Comparative Heatmap",
                 h4("Comparative Heatmap: Models vs. Label"),
                 plotOutput("heatmap_preds", height = "600px")
        )
      )
    )
  )
)

server <- function(input, output, session) {
  
  # === Load data ===
  datos_input <- reactive({
    req(input$datafile)
    ext <- tools::file_ext(input$datafile$name)
    if (ext == "csv") {
      read.csv(input$datafile$datapath, stringsAsFactors = FALSE)
    } else {
      read.delim(input$datafile$datapath, stringsAsFactors = FALSE)
    }
  })
  
  # === Load models ===
  modelos <- reactive({
    req(input$modelfiles)
    mods <- lapply(input$modelfiles$datapath, readRDS)
    names(mods) <- tools::file_path_sans_ext(input$modelfiles$name)
    mods
  })
  
  # === UI updates ===
  output$modelo_sel_ui <- renderUI({
    req(modelos())
    selectInput("modelo_sel", "Select a Model:",
                choices = names(modelos()), selectize = TRUE)
  })
  
  output$id_col_ui <- renderUI({
    req(datos_input())
    choices <- c("None" = "", colnames(datos_input()))
    selectInput("id_col", "Sample ID Column:", choices = choices, selected = "", selectize = TRUE)
  })
  
  output$label_col_ui <- renderUI({
    req(datos_input())
    choices <- c("None" = "", colnames(datos_input()))
    selectInput("label_col", "True Label Column:", choices = choices, selected = "", selectize = TRUE)
  })
  
  # === Helper: prepare data for model ===
  preparar_datos <- function(df, mod) {
    if (inherits(mod, "train") && !is.null(mod$finalModel$xNames)) {
      vars <- mod$finalModel$xNames
    } else if (!is.null(mod$terms)) {
      vars <- attr(mod$terms, "term.labels")
    } else {
      stop("Cannot detect features from the model.")
    }
    faltantes <- setdiff(vars, names(df))
    df2 <- df %>%
      mutate(across(where(is.numeric), as.numeric)) %>%
      add_column(!!!setNames(rep(list(0), length(faltantes)), faltantes))
    df2[, vars, drop = FALSE]
  }
  
  # === Individual prediction ===
  preds <- eventReactive(input$go, {
    req(datos_input(), modelos(), input$modelo_sel)
    df <- datos_input()
    mod <- modelos()[[input$modelo_sel]]
    df_final <- preparar_datos(df, mod)
    pred <- predict(mod, newdata = df_final)
    
    salida <- tibble(Prediction = pred)
    if (input$id_col != "" && input$id_col %in% names(df)) {
      salida <- bind_cols(tibble(ID = df[[input$id_col]]), salida)
    } else {
      salida <- bind_cols(tibble(Sample = 1:nrow(df)), salida)
    }
    if (input$label_col != "" && input$label_col %in% names(df)) {
      salida <- bind_cols(salida, tibble(Label = df[[input$label_col]]))
    }
    salida
  })
  
  # === Predictions for all models ===
  preds_all <- eventReactive(input$go_all, {
    req(datos_input(), modelos())
    df <- datos_input()
    n_obs <- nrow(df)
    res_list <- lapply(names(modelos()), function(mn) {
      mod <- modelos()[[mn]]
      df_final <- preparar_datos(df, mod)
      pred <- tryCatch(predict(mod, newdata = df_final),
                       error = function(e) { rep(NA, n_obs) })
      if (length(pred) != n_obs) {
        pred <- rep_len(pred, n_obs)
      }
      tibble(!!mn := pred)
    })
    tabla <- bind_cols(res_list)
    if (input$id_col != "" && input$id_col %in% names(df)) {
      tabla <- bind_cols(tibble(ID = df[[input$id_col]]), tabla)
    } else {
      tabla <- bind_cols(tibble(Sample = 1:n_obs), tabla)
    }
    if (input$label_col != "" && input$label_col %in% names(df)) {
      tabla <- bind_cols(tabla, tibble(Label = df[[input$label_col]]))
    }
    tabla
  })
  
  # === Outputs ===
  output$preview <- renderTable({
    req(input$datafile)
    tryCatch({
      head(datos_input())
    }, error = function(e) {
      data.frame(Message = "No data loaded or invalid file.")
    })
  })
  output$preds <- renderTable({ req(preds()); preds() })
  output$preds_all <- renderTable({ req(preds_all()); preds_all() })
  
  # === Heatmap ===
  heatmap_plot <- reactive({
    req(preds_all())
    df <- preds_all()
    id_col <- if ("ID" %in% names(df)) "ID" else "Sample"
    df_long <- df %>% pivot_longer(-all_of(id_col), names_to = "Model", values_to = "Value")
    
    value_factor <- as.factor(df_long$Value)
    
    plot_obj <- ggplot(df_long, aes(x = Model, y = .data[[id_col]], fill = value_factor)) +
      geom_tile(color = "grey70") +
      theme_minimal(base_size = 14) +
      labs(x = "Model / Label", y = id_col, fill = "Value") +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
    
    # Intenta usar Set3 si hay pocos niveles, sino usa la escala discreta por defecto
    if (length(unique(value_factor)) <= 12) {
      plot_obj <- plot_obj + scale_fill_brewer(palette = "Set3")
    } else {
      plot_obj <- plot_obj + scale_fill_discrete()
    }
    return(plot_obj)
  })
  output$heatmap_preds <- renderPlot({ heatmap_plot() })
  
  # === Download handlers ===
  output$download_preds <- downloadHandler(
    filename = function() paste0("predictions_", input$modelo_sel, "_", Sys.Date(), ".tsv"),
    content = function(file) write_tsv(preds(), file)
  )
  output$download_preds_all <- downloadHandler(
    filename = function() paste0("comparative_predictions_", Sys.Date(), ".tsv"),
    content = function(file) write_tsv(preds_all(), file)
  )
  output$download_heatmap <- downloadHandler(
    filename = function() paste0("comparative_heatmap_", Sys.Date(), ".pdf"),
    content = function(file) {
      ggsave(file, plot = heatmap_plot(), device = "pdf", width = 10, height = 8)
    }
  )
  
  # === Dynamic HTML Report (CORRECCIÓN CLAVE DE 'isolate()' INTEGRADA) ===
  output$download_html_report <- downloadHandler(
    filename = function() paste0("Prediction_Report_", Sys.Date(), ".html"),
    content = function(file) {
      
      # 1. Requiere que las predicciones comparativas existan
      req(preds_all()) 
      
      # 2. **CORRECCIÓN**: Extraer los valores de las funciones reactivas usando isolate()
      # para evitar que rmarkdown::render falle al intentar evaluar funciones de Shiny.
      current_preds_all <- isolate(preds_all())
      
      # 3. Definir la lista de parámetros
      params_list <- list(
        preds = current_preds_all, # Usamos preds_all para 'preds' por seguridad
        preds_all = current_preds_all,
        date = Sys.Date()
      )
      
      # 4. Generar la plantilla Rmd (on the fly)
      temp_rmd <- tempfile(fileext = ".Rmd")
      rmd_lines <- c(
        "---",
        "title: 'Multiple Model Prediction Report'",
        "output: html_document",
        "params:",
        "  preds: NULL",
        "  preds_all: NULL",
        "  date: NULL",
        "---",
        "",
        "```{r setup, include=FALSE}",
        "library(ggplot2); library(dplyr); library(tidyr); library(caret);",
        "knitr::opts_chunk$set(echo=FALSE, warning=FALSE, message=FALSE)",
        "```",
        "",
        "# 🧾 Multiple Model Prediction Report",
        "",
        "**Date:** `r params$date`",
        "",
        "## 🔹 Individual Predictions (Primeras 10)",
        "```{r}",
        "head(params$preds, 10)", 
        "```",
        "",
        "## 🔹 Comparative Predictions (Primeras 10)",
        "```{r}",
        "head(params$preds_all, 10)",
        "```",
        "",
        "## 🔹 Comparative Heatmap",
        "```{r, fig.width=9, fig.height=6}",
        "df <- params$preds_all",
        "id_col <- if ('ID' %in% names(df)) 'ID' else 'Sample'",
        "df_long <- df %>% pivot_longer(-all_of(id_col), names_to='Model', values_to='Value')",
        "ggplot(df_long, aes(x=Model, y=.data[[id_col]], fill=as.factor(Value))) +",
        " geom_tile(color='grey70') + scale_fill_brewer(palette='Set3') +",
        " theme_minimal(base_size=14) +",
        " labs(title='Comparative Heatmap', x='Model / Label', y=id_col) +",
        " theme(axis.text.x=element_text(angle=45, hjust=1))",
        "```",
        "",
        "## 🔹 Confusion Matrices",
        "```{r}",
        "if ('Label' %in% names(params$preds_all)) {",
        "  label_col <- 'Label'",
        "  model_cols <- setdiff(names(params$preds_all), c('Label','ID','Sample'))",
        "  for (m in model_cols) {",
        "    cat('### Model:', m, '\\n')",
        "    tabla <- table(Predicted = factor(params$preds_all[[m]]), True = factor(params$preds_all[[label_col]]))", 
        "    print(tabla)",
        "    cat('\\n')",
        "  }",
        "} else {",
        "  cat('No true labels provided — cannot compute confusion matrices.')",
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

shinyApp(ui, server)