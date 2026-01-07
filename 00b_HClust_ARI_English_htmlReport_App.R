# =====================================================================
# 📊 Hierarchical Clustering Analysis with HTML Report Export (Final)
# =====================================================================

library(shiny)
library(tidyverse)
library(caret)
library(cluster)
library(factoextra)
library(dendextend)
library(readr)
library(MLmetrics)
library(rmarkdown)
library(mclust) # Añadido mclust para Adjusted Rand Index

options(shiny.maxRequestSize = 50 * 1024^2)

# ----------------- User Interface (UI) -----------------
ui <- fluidPage(
  titlePanel("Análisis de Clustering Jerárquico y Optimización de Clusters"),
  sidebarLayout(
    sidebarPanel(
      fileInput("file1", "Cargar su Archivo TSV", accept = ".tsv"),
      numericInput("k_clusters", "Número de Clusters a Usar (K):", value = 3, min = 2),
      textInput("id_col_cluster", "Nombre de la Columna ID:", value = "ID"),
      # Campo para la columna de Ground Truth (para cálculo de ARI)
      textInput("ground_truth_col", "Columna de Ground Truth (Opcional - para ARI):", value = ""),
      actionButton("run_analysis", "Ejecutar Análisis", class = "btn-primary"),
      tags$hr(),
      p(strong("Opciones de Descarga (Datos):")),
      downloadButton("download_cleaned_data", "Descargar Datos Limpios (NO Escalados) (TSV)"),
      tags$hr(),
      p(strong("Opciones de Descarga (Resultados):")),
      downloadButton("download_dendrogram", "Descargar Dendrograma (PDF)"),
      downloadButton("download_pca_plot", "Descargar Gráfico PCA (PDF)"),
      downloadButton("download_k_plot", "Descargar Gráfico K Óptimo (PDF)"),
      downloadButton("download_means_table", "Descargar Variables Definitorias (TSV)"),
      downloadButton("download_html_report", "📄 Descargar Reporte HTML"),
      tags$hr(),
      p(strong("NOTA:")),
      p("Se recomienda ejecutar el análisis primero para ver el K óptimo sugerido y luego ajustar K.")
    ),
    mainPanel(
      tabsetPanel(
        id = "main_tabs",
        tabPanel("Datos y Escalado",
                 h4("Datos Cargados (Primeras Filas)"),
                 tableOutput("head_data"),
                 h4("Resumen de Datos Escalados"),
                 verbatimTextOutput("scaled_summary")
        ),
        tabPanel("K Óptimo (Método Silueta)",
                 plotOutput("optimal_k_plot"),
                 h4("Interpretación de K Óptimo (Validez Interna)"),
                 textOutput("optimal_k_text"),
                 tags$hr(),
                 h4("Adjusted Rand Index (ARI) - Validez Externa"),
                 textOutput("ari_score")
        ),
        tabPanel("Dendrograma",
                 plotOutput("dendrogram_plot", height = "500px")
        ),
        tabPanel("Visualización de Clusters (PCA)",
                 plotOutput("cluster_pca_plot")
        ),
        tabPanel("Variables Definitorias",
                 h4("Variables que Caracterizan Cada Cluster"),
                 p("Las variables se muestran comparando la media de cada cluster con la media global."),
                 tableOutput("cluster_means_table")
        )
      )
    )
  )
)

# ----------------- Server Logic (SERVER) -----------------
server <- function(input, output, session) {
  
  # 1️⃣ Cleaned input data
  full_data_valid_rows <- reactive({
    req(input$file1)
    df <- read_delim(input$file1$datapath, delim = "\t", col_types = cols(),
                     locale = locale(encoding = "UTF-8"))
    df_valid <- df |> drop_na()
    return(df_valid)
  })
  
  # 2️⃣ Clean numeric data for clustering
  cleaned_numeric_data <- reactive({
    df_valid <- full_data_valid_rows()
    id_col_name <- input$id_col_cluster
    gt_col_name <- input$ground_truth_col # Obtener nombre de columna GT
    
    # Columnas a excluir del clustering (ID y Ground Truth)
    cols_to_exclude <- c()
    if (id_col_name != "" && id_col_name %in% names(df_valid)) {
      cols_to_exclude <- c(cols_to_exclude, id_col_name)
    }
    if (gt_col_name != "" && gt_col_name %in% names(df_valid)) {
      cols_to_exclude <- c(cols_to_exclude, gt_col_name) # Excluir columna GT
    }
    
    if (length(cols_to_exclude) > 0) {
      df_numeric <- df_valid |>
        select(-all_of(cols_to_exclude)) |>
        select(where(is.numeric))
    } else {
      df_numeric <- df_valid |> select(where(is.numeric))
    }
    
    if (ncol(df_numeric) < 2) stop("The file must contain at least 2 numeric columns for clustering.")
    
    var_cero <- nearZeroVar(df_numeric, saveMetrics = TRUE)
    zero_var_cols <- rownames(var_cero)[var_cero$zeroVar == TRUE]
    
    if (length(zero_var_cols) > 0) {
      df_final <- df_numeric |> select(-all_of(zero_var_cols))
      message(paste("Zero-variance variables removed:", paste(zero_var_cols, collapse = ", ")))
    } else df_final <- df_numeric
    
    if (id_col_name != "" && id_col_name %in% names(full_data_valid_rows())) {
      rownames(df_final) <- full_data_valid_rows()[[id_col_name]]
    }
    
    return(df_final)
  })
  
  # 2.5️⃣ Extract Ground Truth Labels (if provided)
  ground_truth_labels <- reactive({
    req(input$file1)
    df <- full_data_valid_rows()
    gt_col_name <- input$ground_truth_col
    
    if (gt_col_name != "" && gt_col_name %in% names(df)) {
      # Asegurar que las etiquetas sean tratadas como caracter (string) para la comparación
      return(as.character(df[[gt_col_name]]))
    } else {
      return(NULL) # No se proporcionó Ground Truth
    }
  })
  
  # 3️⃣ Scaled data
  scaled_data <- reactive({
    df_num <- cleaned_numeric_data()
    scale(df_num)
  })
  
  # 4️⃣ Download table (unscaled clean data)
  download_table_fun <- reactive({
    req(input$run_analysis)
    id_col_name <- input$id_col_cluster
    cleaned_df <- cleaned_numeric_data()
    sample_ids <- rownames(cleaned_df)
    
    final_tbl <- cleaned_df |>
      as_tibble() |>
      mutate(!!sym(id_col_name) := sample_ids, .before = 1)
    
    return(final_tbl)
  })
  
  # 5️⃣ UI outputs
  output$head_data <- renderTable({ head(cleaned_numeric_data()) })
  output$scaled_summary <- renderPrint({ summary(scaled_data()) })
  
  # 6️⃣ Optimal number of clusters
  optimal_k_result <- eventReactive(input$run_analysis, {
    fviz_nbclust(scaled_data(), hcut, method = "silhouette", k.max = min(10, nrow(scaled_data()) - 1))
  })
  
  output$optimal_k_plot <- renderPlot({ optimal_k_result() })
  
  # FIXED conversion to numeric
  output$optimal_k_text <- renderText({
    plot_data <- optimal_k_result()$data
    plot_data$clusters <- as.numeric(as.character(plot_data$clusters))
    
    best_k <- plot_data |>
      filter(y == max(y)) |>
      pull(clusters) |>
      max(na.rm = TRUE)
    
    paste(
      "El método de la silueta sugiere que el número óptimo de clusters (K) es:",
      best_k,
      ". Ajuste el valor de K en la barra lateral para visualizar los resultados con este número."
    )
  })
  
  # 7️⃣ Hierarchical clustering
  hclust_model <- reactive({
    req(input$run_analysis)
    dist_mat <- dist(scaled_data(), method = "euclidean")
    hclust(dist_mat, method = "ward.D2")
  })
  
  # 8️⃣ Dendrogram plot
  dendrogram_plot_fun <- function() {
    k <- input$k_clusters
    hclust_model() |>
      as.dendrogram() |>
      set("branches_k_color", k = k) |>
      plot(main = paste("Dendrograma (Ward.D2) con K =", k))
    rect.hclust(hclust_model(), k = k, border = 2:(k+1))
  }
  
  output$dendrogram_plot <- renderPlot({ dendrogram_plot_fun() })
  
  # 9️⃣ Cluster visualization (PCA)
  cluster_pca_plot_fun <- function() {
    k <- input$k_clusters
    cluster_labels <- cutree(hclust_model(), k = k)
    fviz_cluster(list(data = scaled_data(), cluster = cluster_labels),
                 geom = "point", ellipse.type = "norm",
                 palette = "Set1",
                 main = paste("Clusters en el plano PCA (K =", k, ")"))
  }
  
  output$cluster_pca_plot <- renderPlot({ cluster_pca_plot_fun() })
  
  # 🔟 Defining variables
  cluster_means_table_fun <- reactive({
    req(input$run_analysis)
    k <- input$k_clusters
    cluster_labels <- cutree(hclust_model(), k = k)
    df_labeled <- cleaned_numeric_data()
    df_labeled$Cluster <- as.factor(cluster_labels)
    
    cluster_summary <- df_labeled |>
      as_tibble() |>
      group_by(Cluster) |>
      summarise(across(where(is.numeric), mean)) |>
      select(Cluster, everything())
    
    cluster_summary_long <- cluster_summary |>
      pivot_longer(cols = -Cluster, names_to = "Variable", values_to = "Mean") |>
      pivot_wider(names_from = Cluster, values_from = Mean, names_prefix = "Cluster_")
    
    global_means <- cleaned_numeric_data() |>
      as_tibble() |>
      summarise(across(everything(), mean)) |>
      pivot_longer(cols = everything(), names_to = "Variable", values_to = "Global_Mean")
    
    final_table <- global_means |>
      left_join(cluster_summary_long, by = "Variable") |>
      arrange(Variable)
    
    return(final_table)
  })
  
  output$cluster_means_table <- renderTable({ cluster_means_table_fun() })
  
  # 11️⃣ ARI Calculation
  ari_calculation <- eventReactive(input$run_analysis, {
    gt_labels <- ground_truth_labels()
    if (is.null(gt_labels)) {
      return(NULL) # No se puede calcular si no hay Ground Truth
    }
    
    k <- input$k_clusters
    # Etiquetas resultantes del clustering jerárquico (K seleccionado por el usuario)
    cluster_labels <- cutree(hclust_model(), k = k)
    
    # Asegurar que ambos vectores tienen la misma longitud (deberían tenerla si todo está limpio)
    if (length(cluster_labels) != length(gt_labels)) {
      return(NA) # Error en la longitud de los vectores
    }
    
    # Cálculo del ARI usando la función correcta de mclust
    ari_score <- mclust::adjustedRandIndex(cluster_labels, gt_labels)
    
    return(ari_score)
  })
  
  # 12️⃣ ARI Output
  output$ari_score <- renderText({
    score <- ari_calculation()
    if (is.null(score)) {
      return("ARI no se puede calcular: Por favor, especifique la 'Columna de Ground Truth' para medir la validez externa (Rango: -1 a 1).")
    }
    
    if (is.na(score)) {
      return("Falló el cálculo del ARI. Verifique si la columna 'Ground Truth' tiene el mismo número de filas que los datos numéricos.")
    }
    
    score_text <- format(round(score, 4), nsmall = 4)
    
    return(paste("ARI para K =", input$k_clusters, "es:", score_text))
  })
  
  # ----------------- DOWNLOAD HANDLERS -----------------
  
  output$download_cleaned_data <- downloadHandler(
    filename = function() { paste0("datos_limpios_", Sys.Date(), ".tsv") },
    content = function(file) { write.table(download_table_fun(), file, sep="\t", dec=".", row.names=FALSE, quote=FALSE) }
  )
  
  output$download_k_plot <- downloadHandler(
    filename = function() { paste0("k_optimo_", Sys.Date(), ".pdf") },
    content = function(file) { ggsave(file, plot = optimal_k_result(), device = "pdf", width = 10, height = 7) }
  )
  
  output$download_dendrogram <- downloadHandler(
    filename = function() { paste0("dendrograma_", Sys.Date(), ".pdf") },
    content = function(file) { pdf(file, width = 12, height = 8); dendrogram_plot_fun(); dev.off() }
  )
  
  output$download_pca_plot <- downloadHandler(
    filename = function() { paste0("clusters_pca_", Sys.Date(), ".pdf") },
    content = function(file) { ggsave(file, plot = cluster_pca_plot_fun(), device = "pdf", width = 10, height = 7) }
  )
  
  output$download_means_table <- downloadHandler(
    filename = function() { paste0("variables_definitorias_", Sys.Date(), ".tsv") },
    content = function(file) { write.table(cluster_means_table_fun(), file, sep="\t", dec=".", row.names=FALSE, quote=FALSE) }
  )
  
  # 🆕 HTML REPORT (RMarkdown) - FIX: Removed envir argument
  output$download_html_report <- downloadHandler(
    filename = function() { paste0("Reporte_Clustering_Jerarquico_", Sys.Date(), ".html") },
    content = function(file) {
      temp_rmd <- tempfile(fileext = ".Rmd")
      
      # Obtener el ARI para incluirlo en el reporte
      ari_score_report <- ari_calculation()
      ari_text <- if (is.null(ari_score_report)) {
        "ARI no calculado (columna Ground Truth no especificada)."
      } else if (is.na(ari_score_report)) {
        "Falló el cálculo del ARI."
      } else {
        format(round(ari_score_report, 4), nsmall = 4)
      }
      
      rmd_lines <- c(
        "---",
        "title: 'Reporte de Análisis de Clustering Jerárquico'",
        "output: html_document",
        "params:",
        "  scaled_data: NULL",
        "  cleaned_data: NULL",
        "  optimal_k_plot: NULL",
        "  dendrogram_fun: NULL",
        "  pca_plot_fun: NULL",
        "  cluster_table: NULL",
        "  best_k: NA",
        "  ari_score: NA",
        "  date: NA",
        "---",
        "",
        "```{r setup, include=FALSE}",
        "library(ggplot2)",
        "library(dplyr)",
        "library(factoextra)",
        "library(gridExtra)",
        "library(mclust)",
        "knitr::opts_chunk$set(echo = FALSE, warning = FALSE, message = FALSE)",
        "```",
        "",
        "# 📊 Reporte de Análisis de Clustering Jerárquico",
        "",
        "**Fecha:** `r params$date` ",
        "",
        "**Número óptimo de clusters sugerido (K):** `r params$best_k`",
        "",
        "**Adjusted Rand Index (ARI):** `r params$ari_score`",
        "",
        "## 🔹 K Óptimo (Método Silueta)",
        "```{r}",
        "print(params$optimal_k_plot)",
        "```",
        "",
        "## 🔹 Dendrograma (Ward.D2)",
        "```{r}",
        "params$dendrogram_fun()",
        "```",
        "",
        "## 🔹 Visualización de Clusters (PCA)",
        "```{r}",
        "params$pca_plot_fun()",
        "```",
        "",
        "## 🔹 Variables Definitorias por Cluster",
        "```{r}",
        "head(params$cluster_table, 20)",
        "```"
      )
      writeLines(rmd_lines, temp_rmd)
      
      # Compute best K to include in the report
      plot_data <- optimal_k_result()$data
      plot_data$clusters <- as.numeric(as.character(plot_data$clusters))
      best_k <- plot_data |> filter(y == max(y)) |> pull(clusters) |> max(na.rm = TRUE)
      
      rmarkdown::render(
        input = temp_rmd,
        output_file = file,
        params = list(
          scaled_data = scaled_data(),
          cleaned_data = cleaned_numeric_data(),
          optimal_k_plot = optimal_k_result(),
          dendrogram_fun = dendrogram_plot_fun,
          pca_plot_fun = cluster_pca_plot_fun,
          cluster_table = cluster_means_table_fun(),
          best_k = best_k,
          ari_score = ari_text,
          date = Sys.Date()
        )
      )
    }
  )
}

# ----------------- Run App -----------------
shinyApp(ui, server)
