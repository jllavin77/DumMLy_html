# app.R - Final Version with Scaling Normalization + HTML RMarkdown Report
library(shiny)
library(factoextra)
library(randomcoloR)
library(caret)
library(rgl)
library(htmlwidgets)
library(dplyr)
library(tidyr)
library(ggplot2)
library(gridExtra)
library(base64enc)
library(rmarkdown)

# Set option for rgl not to use the physical display (IMPORTANT)
options(rgl.useNULL = TRUE) 
options(shiny.maxRequestSize = 50 * 1024^2)

# ==============================================================================
# NEW Function: Scaling Normalization (CPM) 🔬
# ==============================================================================
normalizar_por_libreria <- function(df_num, constante = 1e6) {
  df_num <- df_num %>% mutate(across(everything(), as.numeric))
  library_sizes <- rowSums(df_num, na.rm = TRUE)
  df_normalizado <- apply(df_num, 1, function(row) {
    if (sum(row, na.rm = TRUE) > 0) {
      (row / sum(row, na.rm = TRUE)) * constante
    } else {
      rep(0, length(row))
    }
  })
  df_normalizado <- as.data.frame(t(df_normalizado))
  colnames(df_normalizado) <- colnames(df_num)
  return(df_normalizado)
}

# ==============================================================================
# Function: select variables by contribution
# ==============================================================================
seleccionar_variables_por_contribucion <- function(pca_obj, df_original,
                                                   varianza_objetivo = 0.9,
                                                   contrib_umbral = 0.9,
                                                   col_id = 1,
                                                   col_cat = 2) {
  var_explicada <- summary(pca_obj)$importance[3, ]
  num_dim_opt <- which(var_explicada >= varianza_objetivo)[1]
  if (is.na(num_dim_opt)) num_dim_opt <- length(var_explicada)
  
  res.var <- get_pca_var(pca_obj)
  contrib <- res.var$contrib[, 1:num_dim_opt, drop = FALSE]
  contrib_total <- rowSums(contrib)
  contrib_total_norm <- if (sum(contrib_total) == 0) contrib_total else contrib_total / sum(contrib_total)
  
  cutoff <- if (length(contrib_total_norm)>0) contrib_umbral / length(contrib_total_norm) else 1
  selected_vars <- names(contrib_total_norm[contrib_total_norm >= cutoff])
  excluded_vars <- setdiff(names(contrib_total_norm), selected_vars)
  
  vars_presentes <- intersect(selected_vars, colnames(df_original))
  if (length(vars_presentes) == 0) {
    df_filtrado <- df_original[, c(col_id, col_cat), drop = FALSE]
  } else {
    cols_to_take <- c(col_id, col_cat, match(vars_presentes, colnames(df_original)))
    df_filtrado <- df_original[, cols_to_take, drop = FALSE]
  }
  
  contrib_df <- data.frame(
    variable = names(contrib_total_norm),
    contrib = as.numeric(contrib_total_norm),
    stringsAsFactors = FALSE
  )
  
  list(
    df_filtrado = df_filtrado,
    selected_vars = vars_presentes,
    excluded_vars = excluded_vars,
    contribuciones = contrib_df,
    num_dim_opt = num_dim_opt,
    var_explicada = var_explicada[num_dim_opt]
  )
}

# ==============================================================================
# UI
# ==============================================================================
ui <- fluidPage(
  tags$head(
    tags$script(HTML("
        Shiny.addCustomMessageHandler('download_rgl_png', function(message) {
          var el = document.getElementById('plot3D');
          if (el && el.rglinstance) {
            el.rglinstance.canvas.toBlob(function(blob) {
              var url = URL.createObjectURL(blob);
              var a = document.createElement('a');
              a.href = url;
              a.download = 'PCA_3D_' + new Date().toISOString().slice(0, 10) + '.png';
              document.body.appendChild(a);
              a.click();
              document.body.removeChild(a);
              URL.revokeObjectURL(url);
            }, 'image/png');
          } else {
            alert('The 3D plot is not ready for download.');
          }
        });
    "))
  ),
  titlePanel("🔮 PCA with Reduction and Deluxe Export (Final)"),
  
  sidebarLayout(
    sidebarPanel(
      fileInput("file", "Upload File (.tsv)", accept = ".tsv"),
      
      uiOutput("colCatUI"), 
      
      sliderInput("umbral", "Contribution Threshold (%)", min = 50, max = 100, value = 80),
      actionButton("run", "Run PCA"),
      hr(),
      downloadButton("downloadReduced", "⬇️ Reduced Dataset (TSV)"),
      downloadButton("downloadEigen", "⬇️ Eigenvalues (TSV)"),
      downloadButton("downloadContrib", "⬇️ Contributions (TSV)"),
      downloadButton("downloadSelVars", "⬇️ Selected Variables (TXT)"),
      downloadButton("downloadExcVars", "⬇️ Excluded Variables (TXT)"),
      downloadButton("downloadPDF", "⬇️ Plots PDF"),
      downloadButton("download3DHTML", "⬇️ Interactive PCA 3D (HTML)"), 
      actionButton("download3DPNG_js", "⬇️ Static PCA 3D (PNG)"),
      hr(),
      downloadButton("downloadRmdReport", "📄 Full PCA Report (HTML)")  # << NUEVO BOTÓN
    ),
    
    mainPanel(
      tabsetPanel(
        tabPanel("Reduced Data Table", tableOutput("tableReduced")),
        tabPanel("Scree Plot", plotOutput("plotScree")),
        tabPanel("PCA Variables", plotOutput("plotVar", height = "600px")),
        tabPanel("PCA 3D", rglwidgetOutput("plot3D"))
      )
    )
  )
)

# ==============================================================================
# SERVER
# ==============================================================================
server <- function(input, output, session) {
  
  dataset_raw <- reactive({
    req(input$file)
    PA_table <- read.delim(input$file$datapath, sep="\t", header=TRUE, stringsAsFactors=FALSE, check.names = FALSE)
    if("X" %in% colnames(PA_table)) PA_table$X <- NULL 
    colnames(PA_table) <- gsub("X\\.", "", colnames(PA_table))
    colnames(PA_table) <- gsub("X", "", colnames(PA_table)) 
    return(PA_table)
  })
  
  output$colCatUI <- renderUI({
    df <- dataset_raw()
    req(df)
    selected_col <- if (ncol(df) >= 2) colnames(df)[2] else colnames(df)[1]
    
    selectInput("col_cat", "Select Category Column:",
                choices = colnames(df), 
                selected = selected_col)
  })
  
  resultados <- eventReactive(input$run, {
    df0 <- dataset_raw()
    req(input$col_cat)
    
    col_cat_name <- input$col_cat
    col_cat_index <- which(colnames(df0) == col_cat_name)
    col_id_index <- 1
    
    cols_para_pca <- setdiff(colnames(df0), c(colnames(df0)[col_id_index], col_cat_name))
    mat_pca_num_raw <- df0[, cols_para_pca, drop = FALSE] %>% 
      mutate(across(where(is.character), ~ suppressWarnings(as.numeric(.))))
    
    mat_pca_num_norm <- normalizar_por_libreria(mat_pca_num_raw)
    mat_pca_num_proc <- mat_pca_num_norm[, sapply(mat_pca_num_norm, function(x) var(x, na.rm=TRUE) > 0), drop = FALSE]
    
    pca <- prcomp(na.omit(mat_pca_num_proc), center = TRUE, scale. = TRUE)
    
    df_original_clean <- df0[, c(col_id_index, col_cat_index), drop = FALSE]
    df_original_clean <- cbind(df_original_clean, mat_pca_num_proc)
    
    resultado_vars <- seleccionar_variables_por_contribucion(
      pca_obj = pca,
      df_original = df_original_clean,
      varianza_objetivo = 0.9,
      contrib_umbral = input$umbral / 100,
      col_id = 1,
      col_cat = 2
    )
    
    list(
      pca = pca,
      eig.val = get_eigenvalue(pca),
      contribuciones = resultado_vars$contribuciones,
      df_reducido = resultado_vars$df_filtrado,
      selected_vars = resultado_vars$selected_vars,
      excluded_vars = resultado_vars$excluded_vars,
      categorias = df_original_clean[[2]]
    )
  })
  
  output$tableReduced <- renderTable({ req(resultados()); head(resultados()$df_reducido, 20) })
  output$plotScree <- renderPlot({ req(resultados()); fviz_eig(resultados()$pca) })
  output$plotVar <- renderPlot({
    req(resultados())
    fviz_pca_var(resultados()$pca, col.var = "contrib", gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"), repel = TRUE)
  })
  
  output$plot3D <- renderRglwidget({
    req(resultados())
    pca <- resultados()$pca
    categorias <- resultados()$categorias
    levs <- levels(as.factor(categorias))
    group.col <- distinctColorPalette(length(levs))
    
    rgl::open3d(useNULL = TRUE)
    ddd <- as.matrix(pca$x[, 1:3])
    plot3d(ddd, col = group.col[as.numeric(as.factor(categorias))], type = "s", size = 1, axes = FALSE)
    axes3d(edges = c("x--", "y--", "z"), lwd = 3, axes.len = 2, labels = FALSE)
    legend3d("right", legend = levs, pch = 16, col = group.col, cex = 0.9, inset = c(0.01, 0))
    grid3d("x"); grid3d("y"); grid3d("z")
    w <- rglwidget()
    rgl::close3d()
    return(w)
  })
  
  # ----------------------------------------------------------------------------------
  # DOWNLOAD HANDLERS
  # ----------------------------------------------------------------------------------
  output$downloadReduced <- downloadHandler( filename = function() { "PCA_reduced_dataset.tsv" }, content = function(file) { write.table(resultados()$df_reducido, file, sep="\t", quote=FALSE, row.names=FALSE) } )
  output$downloadEigen <- downloadHandler( filename = function() { "PCA_eigenvalues.tsv" }, content = function(file) { write.table(resultados()$eig.val, file, sep="\t", quote=FALSE, row.names=FALSE) } )
  output$downloadContrib <- downloadHandler( filename = function() { "PCA_variables_contributions.tsv" }, content = function(file) { write.table(resultados()$contribuciones, file, sep="\t", quote=FALSE, row.names=FALSE) } )
  output$downloadSelVars <- downloadHandler( filename = function() { "PCA_selected_variables.txt" }, content = function(file) { writeLines(resultados()$selected_vars, file) } )
  output$downloadExcVars <- downloadHandler( filename = function() { "PCA_excluded_variables.txt" }, content = function(file) { writeLines(resultados()$excluded_vars, file) } )
  output$downloadPDF <- downloadHandler(
    filename = function() { "PCA_plots.pdf" },
    content = function(file) { pdf(file); print(fviz_eig(resultados()$pca)); print(fviz_pca_var(resultados()$pca, col.var="contrib", gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"), repel = TRUE)); dev.off() }
  )
  
  output$download3DHTML <- downloadHandler(
    filename = function() { "PCA_3D_interactive.html" },
    content = function(file) {
      req(resultados())
      pca <- resultados()$pca
      categorias <- resultados()$categorias
      levs <- levels(as.factor(categorias))
      group.col <- distinctColorPalette(length(levs))
      
      rgl::open3d(useNULL = TRUE) 
      ddd <- as.matrix(pca$x[, 1:3])
      plot3d(ddd, col = group.col[as.numeric(as.factor(categorias))], type = "s", size = 1, axes = FALSE)
      axes3d(edges = c("x--","y--","z"), lwd = 3, labels = FALSE)
      legend3d("right", legend = levs, pch = 16, col = group.col, cex = 0.9, inset = c(0.01, 0))
      grid3d("x"); grid3d("y"); grid3d("z")
      
      tmp_html <- tempfile(fileext = ".html")
      htmlwidgets::saveWidget(rgl::rglwidget(), tmp_html, selfcontained = TRUE)
      file.copy(tmp_html, file, overwrite = TRUE)
      rgl::close3d() 
    },
    contentType = "text/html" 
  )
  
  observeEvent(input$download3DPNG_js, {
    session$sendCustomMessage(type = 'download_rgl_png', message = list())
  })
  
  # ==============================================================================
  # 📄 DOWNLOAD PCA REPORT (HTML RMarkdown)
  # ==============================================================================
  output$downloadRmdReport <- downloadHandler(
    filename = function() { "PCA_full_report.html" },
    content = function(file) {
      req(resultados())
      res <- resultados()
      
      temp_rmd <- file.path(tempdir(), "PCA_report.Rmd")
      rmd_lines <- c(
        "---",
        "title: 'PCA Analysis Report'",
        "output:",
        "  html_document:",
        "    toc: true",
        "    toc_float: true",
        "params:",
        "  res: NULL",
        "  date: NA",
        "---",
        "",
        "```{r setup, include=FALSE}",
        "library(ggplot2); library(dplyr); library(factoextra); library(rgl); library(htmlwidgets); library(randomcoloR)",
        "knitr::opts_chunk$set(echo=FALSE, warning=FALSE, message=FALSE, fig.width=8, fig.height=6)",
        "```",
        "",
        "# 🔮 PCA Analysis Report",
        "",
        "**Date of Analysis:** `r params$date`  ",
        "",
        "## 1️⃣ Scree Plot — Variance Explained",
        "```{r}",
        "fviz_eig(params$res$pca)",
        "```",
        "",
        "## 2️⃣ PCA Variables Plot — Contributions and Correlations",
        "```{r}",
        "fviz_pca_var(params$res$pca, col.var='contrib', gradient.cols=c('#00AFBB', '#E7B800', '#FC4E07'), repel=TRUE)",
        "```",
        "",
        "## 3️⃣ Top 20 Variable Contributions",
        "```{r}",
        "head(params$res$contribuciones, 20)",
        "```",
        "",
        "## 4️⃣ Reduced Dataset (First 20 Rows)",
        "```{r}",
        "head(params$res$df_reducido, 20)",
        "```",
        "",
        "## 5️⃣ Interactive 3D PCA Plot",
        "```{r, results='asis'}",
        "pca <- params$res$pca",
        "categorias <- params$res$categorias",
        "levs <- levels(as.factor(categorias))",
        "group.col <- randomcoloR::distinctColorPalette(length(levs))",
        "rgl::open3d(useNULL = TRUE)",
        "ddd <- as.matrix(pca$x[,1:3])",
        "plot3d(ddd, col=group.col[as.numeric(as.factor(categorias))], type='s', size=1, axes=FALSE)",
        "axes3d(edges=c('x--','y--','z'), lwd=3, labels=FALSE)",
        "legend3d('right', legend=levs, pch=16, col=group.col, cex=0.9, inset=c(0.01,0))",
        "grid3d('x'); grid3d('y'); grid3d('z')",
        "w <- rglwidget()",
        "rgl::close3d()",
        "w",
        "```"
      )
      
      writeLines(rmd_lines, temp_rmd)
      rmarkdown::render(
        input = temp_rmd,
        output_file = file,
        params = list(res = res, date = Sys.Date()),
        envir = new.env(parent = globalenv())
      )
    }
  )
}

# ==============================================================================
# Run App
# ==============================================================================
shinyApp(ui, server)

