# Shiny app with embedded RMarkdown report
# Save this file as app.R and run with: shiny::runApp('.')

# --- Packages ---------------------------------------------------------------
required <- c('shiny','shinythemes','DT','ggplot2','dplyr','tidyr','randomForest',
              'ranger','caret','rmarkdown','knitr','shinycssloaders','vip')
new <- required[!(required %in% installed.packages()[,'Package'])]
if(length(new)) message('Please install missing packages: ', paste(new, collapse=', '))

library(shiny)
library(shinythemes)
library(DT)
library(ggplot2)
library(dplyr)
library(tidyr)
library(randomForest)
library(ranger)
library(caret)
library(rmarkdown)
library(knitr)
library(shinycssloaders)
library(vip)

options(shiny.maxRequestSize = 50 * 1024^2) # Aumentar límite de archivo

# --- Embedded RMarkdown template (string) ---------------------------------
# Esta plantilla espera recibir los resultados del modelo directamente como parámetros.
rmd_template <- '---
title: "Reporte de Análisis Ambiental y Predictivo"
output:
  html_document:
    toc: true
    toc_depth: 3
    theme: flatly
params:
  lm_summary: NULL
  rf_performance: NULL
  vip_plot_fun: NULL
  correlations: NULL
  date: NULL
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo=FALSE, warning=FALSE, message=FALSE)
library(ggplot2)
library(dplyr)
library(vip)
```

# 📈 Reporte de Análisis Ambiental

**Fecha del Reporte:** `r params$date`

Este reporte presenta los resultados del análisis predictivo de la variable **Nota** (o **Categoría**) utilizando variables de calidad ambiental. Los modelos ejecutados incluyen Regresión Lineal (Stepwise) y Random Forest.

## 1. Correlaciones con la Nota

Se muestra la correlación de Pearson entre las variables numéricas y la variable objetivo "Nota" (ordenadas por el valor absoluto de correlación).

```{r correlations}
cor_df <- data.frame(Correlacion = params$correlations)
cor_df <- tibble::rownames_to_column(cor_df, "Variable") %>%
  filter(Variable != "Nota") %>%
  arrange(desc(abs(Correlacion)))

knitr::kable(head(cor_df, 15), caption = "Top 15 Correlaciones con la Nota")
```

## 2. Resultados de Modelado Predictivo

### 2.1 Regresión Lineal (Stepwise)

Se utilizó un modelo de regresión lineal con selección de variables Stepwise.

```{r linear_model_summary}
print(params$lm_summary)
```

### 2.2 Random Forest Performance

Métricas de desempeño del modelo Random Forest (calculadas en el conjunto de prueba):

```{r rf_performance_table}
perf_df <- data.frame(Métrica = names(params$rf_performance), Valor = params$rf_performance)
knitr::kable(perf_df, caption = "Métricas de Desempeño del Random Forest (Test Set)")
```

### 2.3 Importancia de Variables (Random Forest)

Este gráfico muestra la importancia relativa de cada variable de entrada en la predicción de la **Nota**.

```{r vip_plot, fig.height=8, fig.width=10}
params$vip_plot_fun()
```

## 3. Conclusiones

Este reporte sirve como base para identificar qué variables ambientales tienen mayor impacto estadístico y predictivo sobre la Nota/Categoría.
'

# --- Shiny UI ---------------------------------------------------------------
ui <- fluidPage(
  theme = shinytheme('flatly'),
  titlePanel('Análisis: influencia de variables ambientales sobre Nota/Categoría'),
  sidebarLayout(
    sidebarPanel(
      fileInput('file', 'Sube un CSV (o usa el botón de ejemplo)', accept = c('.csv')),
      actionButton('load_example', 'Cargar ejemplo (primeras filas provistas)'),
      hr(),
      selectInput('target', 'Variable objetivo para modelado', choices = c('Nota','Categoria'), selected = 'Nota'),
      selectInput('vent_var','Variable de ventilación', choices = c('Ventilacion'), selected = 'Ventilacion'),
      hr(),
      actionButton('run_models', 'Ejecutar modelos', class = "btn-primary"),
      downloadButton('download_report', '📄 Descargar reporte HTML'), # Nuevo patrón de descarga
      width = 3
    ),
    mainPanel(
      tabsetPanel(
        id = "main_tabs",
        tabPanel('Vista datos', DTOutput('table') %>% shinycssloaders::withSpinner()),
        tabPanel('EDA',
                 fluidRow(column(6, plotOutput('hist_target') %>% shinycssloaders::withSpinner()),
                          column(6, plotOutput('box_by_vent') %>% shinycssloaders::withSpinner()))),
        tabPanel('Modelos',
                 h4("Sumario de Modelos (Regresión y RF)"),
                 verbatimTextOutput('model_summary') %>% shinycssloaders::withSpinner(),
                 h4("Importancia de Variables (Random Forest)"),
                 plotOutput('vip_plot') %>% shinycssloaders::withSpinner()),
        tabPanel('Notas',
                 h4('Instrucciones'),
                 p('Sube un CSV con las columnas indicadas. El reporte HTML descargable utiliza los objetos y resultados generados al pulsar "Ejecutar modelos".')
        )
      )
    )
  )
)

# --- Shiny Server -----------------------------------------------------------
server <- function(input, output, session){
  # Reactive dataset
  data_reactive <- reactiveVal(NULL)
  
  observeEvent(input$file, {
    req(input$file)
    df <- read.csv(input$file$datapath, stringsAsFactors = FALSE)
    data_reactive(df)
  })
  
  observeEvent(input$load_example, {
    example_text <- "SensorID,Ventilacion,Nota,Categoria,Eval3_media_total_co_corregido_max_diario,Eval3_media_total_co_corregido_min_diario,Eval3_media_total_co2_corregido_media,Eval3_media_total_co2_corregido_sd,Eval3_media_total_co2_corregido_max_diario,Eval3_media_total_co2_corregido_min_diario,Eval3_media_total_nh3_corregido_media\n11_AMASA_3_2_1_30_07_25,ELECTRICO,165,EXCELENTE,0.0357,0,932.6166,176.7275,1354.407,638.6857,2.7834\n12_AMASA_3_2_2_30_07_25,ELECTRICO,160,EXCELENTE,0.5704,0,1080.747,164.7848,1470.035,745.5986,4.7795\n13_villa_3_2_1_06_08_25,natural,150,EXCELENTE,0,0,1779.162,398.3607,2706.929,947.4143,5.71198\n14_villa_3_2_2_06_08_25,natural,150,EXCELENTE,0.6143,0.6,1823.343,424.7721,2645.671,945.2,5.84586\n15_amezketa_3_2_1_20_08_25,ELECTRICO,120,BUENO,0,0,1419.027,442.7078,2260.246,694.8651,4.313599\n16_amezketa_3_2_3_20_08_25,natural,115,BUENO,0.0357,0,1149.521,246.1098,1650.493,682.1714,1.62201"
    tmpf <- tempfile(fileext = '.csv')
    writeLines(example_text, tmpf)
    df <- read.csv(tmpf, stringsAsFactors = FALSE)
    data_reactive(df)
  })
  
  output$table <- renderDT({
    req(data_reactive())
    dat <- data_reactive()
    datatable(dat, options = list(pageLength = 10, scrollX = TRUE))
  })
  
  output$hist_target <- renderPlot({
    req(data_reactive())
    dat <- data_reactive()
    if(input$target == 'Nota' && 'Nota' %in% names(dat)){
      ggplot(dat, aes(x = as.numeric(Nota))) + geom_histogram(bins = 20) + ggtitle('Distribución de Nota')
    } else if(input$target == 'Categoria' && 'Categoria' %in% names(dat)){
      ggplot(dat, aes(x = Categoria)) + geom_bar() + ggtitle('Distribución de Categoria')
    }
  })
  
  output$box_by_vent <- renderPlot({
    req(data_reactive())
    dat <- data_reactive()
    if('Nota' %in% names(dat) & 'Ventilacion' %in% names(dat)){
      ggplot(dat, aes(x = as.factor(Ventilacion), y = as.numeric(Nota))) + geom_boxplot() + ggtitle('Nota por Ventilación')
    }
  })
  
  # Models & Report Objects
  model_res <- eventReactive(input$run_models, {
    dat <- data_reactive()
    req(dat)
    
    if(input$target == 'Nota'){
      mod_df <- dat %>% select(where(function(x) is.numeric(x) || is.character(x) || is.factor(x)))
      
      mod_df <- mod_df %>% mutate_if(is.character, as.factor)
      if('SensorID' %in% names(mod_df)) mod_df$SensorID <- NULL
      
      mod_df <- na.omit(mod_df)
      if(nrow(mod_df) < 5) return(list(error='No hay suficientes filas limpias para modelar'))
      
      mod_df$Nota <- as.numeric(mod_df$Nota)
      
      if(length(unique(mod_df$Nota)) < 2) return(list(error='Nota no varía lo suficiente para modelar.'))
      
      set.seed(123)
      train_idx <- createDataPartition(mod_df$Nota, p = .8, list = FALSE)
      train <- mod_df[train_idx,]
      test <- mod_df[-train_idx,]
      
      # 1. Linear model (stepwise)
      lm0 <- lm(Nota ~ ., data = train)
      lm_step <- step(lm0, trace = 0)
      lm_summary_obj <- summary(lm_step) # Objeto sumario para el reporte
      
      # 2. Random forest
      rf <- randomForest(Nota ~ ., data = train, ntree = 500)
      pred <- predict(rf, newdata = test)
      perf <- postResample(pred = pred, obs = test$Nota)
      
      # 3. VIP Plot function (para pasar al RMD)
      vip_plot_fun <- function() {
        vip::vip(rf, num_features = 15, bar_width = 0.8) +
          labs(title = "Importancia de Variables (Random Forest)")
      }
      
      # 4. Correlaciones (para pasar al RMD)
      nums_all <- dat %>% select_if(is.numeric)
      cor_nota <- cor(nums_all, use = "pairwise.complete.obs")["Nota",]
      
      # Retornar resultados
      list(
        lm_step = lm_step,
        lm_summary_obj = lm_summary_obj, # Objeto para reporte
        rf = rf,
        perf = perf,
        vip_plot_fun = vip_plot_fun, # Función para reporte
        correlations = cor_nota # Vector de correlaciones
      )
      
    } else {
      return(list(error = 'Modelado para Categoria no implementado en este demo.'))
    }
  })
  
  # UI Outputs
  output$model_summary <- renderPrint({
    res <- model_res()
    if(is.null(res) || !is.null(res$error)) {
      if (!is.null(res$error)) return(res$error)
      return('Pulse "Ejecutar modelos"')
    }
    
    cat('--- Regresión Lineal (Stepwise) ---\\n')
    print(summary(res$lm_step))
    cat('\n--- Random Forest Performance (Test) ---\\n')
    print(res$perf)
  })
  
  output$vip_plot <- renderPlot({
    res <- model_res()
    req(res)
    if(!is.null(res$error)) return()
    res$vip_plot_fun() # Llamamos a la función
  })
  
  # Report generation (siguiendo el patrón del código de clustering)
  output$download_report <- downloadHandler(
    filename = function(){
      paste0('reporte_analisis_ambiental_', Sys.Date(), '.html')
    },
    content = function(file){
      res <- model_res()
      req(res) # Asegurar que los modelos se hayan ejecutado
      
      temp_rmd <- tempfile(fileext = ".Rmd")
      writeLines(rmd_template, temp_rmd) # Escribir la plantilla
      
      # Renderizar con los parámetros de resultados
      rmarkdown::render(
        input = temp_rmd, 
        output_file = file,
        params = list(
          lm_summary = res$lm_summary_obj,
          rf_performance = res$perf,
          vip_plot_fun = res$vip_plot_fun,
          correlations = res$correlations,
          date = Sys.Date()
        ), 
        envir = new.env(parent = globalenv()) 
      )
    },
    contentType = 'text/html'
  )
}

# Run the app
shinyApp(ui, server)