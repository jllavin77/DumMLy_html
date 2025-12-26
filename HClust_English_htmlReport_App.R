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

options(shiny.maxRequestSize = 50 * 1024^2)

# ----------------- User Interface (UI) -----------------
ui <- fluidPage(
  titlePanel("Hierarchical Clustering Analysis and Cluster Optimization"),
  sidebarLayout(
    sidebarPanel(
      fileInput("file1", "Upload your TSV File", accept = ".tsv"),
      numericInput("k_clusters", "Number of Clusters to Use (K):", value = 3, min = 2),
      textInput("id_col_cluster", "ID Column Name:", value = "ID"),
      actionButton("run_analysis", "Run Analysis", class = "btn-primary"),
      tags$hr(),
      p(strong("Download Options (Data):")),
      downloadButton("download_cleaned_data", "Download Clean Data (NOT Scaled) (TSV)"),
      tags$hr(),
      p(strong("Download Options (Results):")),
      downloadButton("download_dendrogram", "Download Dendrogram (PDF)"),
      downloadButton("download_pca_plot", "Download PCA Plot (PDF)"),
      downloadButton("download_k_plot", "Download Optimal K Plot (PDF)"),
      downloadButton("download_means_table", "Download Defining Variables (TSV)"),
      downloadButton("download_html_report", "📄 Download HTML Report"),  # 🆕
      tags$hr(),
      p(strong("NOTE:")),
      p("It is recommended to run the analysis first to see the suggested optimal K and then adjust K.")
    ),
    mainPanel(
      tabsetPanel(
        id = "main_tabs",
        tabPanel("Data and Scaling",
                 h4("Loaded Data (First Rows)"),
                 tableOutput("head_data"),
                 h4("Summary of Scaled Data"),
                 verbatimTextOutput("scaled_summary")
        ),
        tabPanel("Optimal K (Silhouette Method)",
                 plotOutput("optimal_k_plot"),
                 h4("Optimal K Interpretation"),
                 textOutput("optimal_k_text")
        ),
        tabPanel("Dendrogram",
                 plotOutput("dendrogram_plot", height = "500px")
        ),
        tabPanel("Cluster Visualization (PCA)",
                 plotOutput("cluster_pca_plot")
        ),
        tabPanel("Defining Variables",
                 h4("Variables Characterizing Each Cluster"),
                 p("Variables are shown by comparing the mean of each cluster to the global mean."),
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
    
    if (id_col_name != "" && id_col_name %in% names(df_valid)) {
      df_numeric <- df_valid |> 
        select(-!!sym(id_col_name)) |>
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
  
  # ✅ FIXED conversion to numeric
  output$optimal_k_text <- renderText({
    plot_data <- optimal_k_result()$data
    plot_data$clusters <- as.numeric(as.character(plot_data$clusters))
    
    best_k <- plot_data |> 
      filter(y == max(y)) |> 
      pull(clusters) |> 
      max(na.rm = TRUE)
    
    paste(
      "The silhouette method suggests that the optimal number of clusters (K) is:",
      best_k,
      ". Adjust the K value in the sidebar to visualize the results with this number."
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
      plot(main = paste("Dendrogram (Ward.D2) with K =", k)) 
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
                 main = paste("Clusters on PCA plane (K =", k, ")"))
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
  
  # ----------------- DOWNLOAD HANDLERS -----------------
  
  output$download_cleaned_data <- downloadHandler(
    filename = function() { paste0("cleaned_data_", Sys.Date(), ".tsv") },
    content = function(file) { write.table(download_table_fun(), file, sep="\t", dec=".", row.names=FALSE, quote=FALSE) }
  )
  
  output$download_k_plot <- downloadHandler(
    filename = function() { paste0("optimal_k_", Sys.Date(), ".pdf") },
    content = function(file) { ggsave(file, plot = optimal_k_result(), device = "pdf", width = 10, height = 7) }
  )
  
  output$download_dendrogram <- downloadHandler(
    filename = function() { paste0("dendrogram_", Sys.Date(), ".pdf") },
    content = function(file) { pdf(file, width = 12, height = 8); dendrogram_plot_fun(); dev.off() }
  )
  
  output$download_pca_plot <- downloadHandler(
    filename = function() { paste0("clusters_pca_", Sys.Date(), ".pdf") },
    content = function(file) { ggsave(file, plot = cluster_pca_plot_fun(), device = "pdf", width = 10, height = 7) }
  )
  
  output$download_means_table <- downloadHandler(
    filename = function() { paste0("defining_variables_", Sys.Date(), ".tsv") },
    content = function(file) { write.table(cluster_means_table_fun(), file, sep="\t", dec=".", row.names=FALSE, quote=FALSE) }
  )
  
  # 🆕 HTML REPORT (RMarkdown)
  output$download_html_report <- downloadHandler(
    filename = function() { paste0("Hierarchical_Clustering_Report_", Sys.Date(), ".html") },
    content = function(file) {
      temp_rmd <- tempfile(fileext = ".Rmd")
      rmd_lines <- c(
        "---",
        "title: 'Hierarchical Clustering Analysis Report'",
        "output: html_document",
        "params:",
        "  scaled_data: NULL",
        "  cleaned_data: NULL",
        "  optimal_k_plot: NULL",
        "  dendrogram_fun: NULL",
        "  pca_plot_fun: NULL",
        "  cluster_table: NULL",
        "  best_k: NA",
        "  date: NA",
        "---",
        "",
        "```{r setup, include=FALSE}",
        "library(ggplot2)",
        "library(dplyr)",
        "library(factoextra)",
        "library(gridExtra)",
        "knitr::opts_chunk$set(echo = FALSE, warning = FALSE, message = FALSE)",
        "```",
        "",
        "# 📊 Hierarchical Clustering Analysis Report",
        "",
        "**Date:** `r params$date`  ",
        "",
        "**Suggested optimal number of clusters (K):** `r params$best_k`",
        "",
        "## 🔹 Optimal K (Silhouette Method)",
        "```{r}",
        "print(params$optimal_k_plot)",
        "```",
        "",
        "## 🔹 Dendrogram (Ward.D2)",
        "```{r}",
        "params$dendrogram_fun()",
        "```",
        "",
        "## 🔹 Cluster Visualization (PCA)",
        "```{r}",
        "params$pca_plot_fun()",
        "```",
        "",
        "## 🔹 Defining Variables per Cluster",
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
          date = Sys.Date()
        ),
        envir = new.env(parent = globalenv())
      )
    }
  )
}

# ----------------- Run App -----------------
shinyApp(ui, server)

