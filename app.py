library(shiny)
library(bslib)
library(tidyverse)
library(tidyquant)
library(DT)
library(openxlsx)
library(httr)
library(rvest)
library(lubridate)
library(bsicons)

# =========================
# 🔑 API KEY
# =========================
FINNHUB_API_KEY <- "d54rt91r01qojbih3rd0d54rt91r01qojbih3rdg"

# =========================
# 🎨 THEME
# =========================
my_theme <- bs_theme(
  bootswatch = "darkly",
  primary = "#00bc8c",
  base_font = font_google("Inter")
)

# =========================
# 🧠 HELPER FUNCTIONS
# =========================

# 1. Fast: Get Next Earnings Date
get_next_earnings <- function(ticker) {
  tryCatch({
    url <- paste0("https://finance.yahoo.com/quote/", ticker)
    page <- session(url, user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
    text_content <- page %>% html_text()
    date_match <- str_extract(text_content, "[A-Z][a-z]{2}\\s+\\d{1,2},\\s+\\d{4}")
    
    if (!is.na(date_match)) {
      date_obj <- mdy(date_match)
      if (date_obj >= Sys.Date()) return(format(date_obj, "%Y-%m-%d"))
    }
    return("TBD")
  }, error = function(e) return("TBD"))
}

# 2. Fast: Get Current Price Only
get_current_price <- function(ticker) {
  tryCatch({
    # We use a lighter call just for the last price
    quote <- tq_get(ticker, get = "stock.prices", from = Sys.Date() - days(5))
    if(nrow(quote) > 0) return(tail(quote$close, 1))
    return(NA)
  }, error = function(e) return(NA))
}

# 3. Slow/Heavy: Get Detailed History (Only runs on click)
get_history_analysis <- function(ticker) {
  # A. Fetch Past Earnings Events
  url <- paste0("https://finnhub.io/api/v1/stock/earnings?symbol=", ticker, "&token=", FINNHUB_API_KEY)
  res <- tryCatch(GET(url), error = function(e) NULL)
  
  if (is.null(res)) return(NULL)
  
  data <- content(res, as = "parsed")
  if (length(data) == 0) return(NULL)
  
  earnings_df <- map_dfr(data, ~as_tibble(compact(.x))) %>% slice(1:4)
  
  # B. Fetch 2 Years of Price History for Calculations
  prices <- tq_get(ticker, get = "stock.prices", from = Sys.Date() - years(2))
  
  # C. Calculate Reactions
  calc_move <- function(date, days) {
    tryCatch({
      target <- as.Date(date)
      pre <- prices %>% filter(date <= target) %>% tail(1) %>% pull(close)
      post <- prices %>% filter(date >= target + days(1)) %>% head(days) %>% tail(1) %>% pull(close)
      if(length(pre)==0 || length(post)==0) return(NA)
      return((post - pre) / pre)
    }, error = function(e) NA)
  }
  
  earnings_df %>%
    mutate(
      `Event Date` = period,
      `Surprise` = surprise,
      `1D Move` = map_dbl(period, ~calc_move(.x, 1)),
      `3D Move` = map_dbl(period, ~calc_move(.x, 3))
    ) %>%
    select(`Event Date`, `Actual` = actual, `Estimate` = estimate, `Surprise`, `1D Move`, `3D Move`)
}

# =========================
# 🖥️ UI
# =========================
ui <- page_sidebar(
  theme = my_theme,
  title = "📡 Earnings Radar (Fast Mode)",
  
  sidebar = sidebar(
    title = "Portfolio",
    fileInput("file_upload", "📂 Upload File", accept = c(".csv", ".xlsx")),
    # DEFAULT IS NOW EMPTY to prevent slow start
    textAreaInput("manual_tickers", "📝 Add Tickers", value = "", placeholder = "AAPL\nTSLA", height = "100px"),
    actionButton("run_scan", "🚀 Scan Market", class = "btn-primary w-100"),
    hr(),
    helpText("1. Scan creates a summary list."),
    helpText("2. Click 'Analyze' to see history.")
  ),
  
  card(
    full_screen = TRUE,
    card_header("🎯 Market Summary"),
    DTOutput("summary_table")
  )
)

# =========================
# ⚙️ SERVER
# =========================
server <- function(input, output, session) {
  
  # Store the summary data here
  summary_data <- reactiveVal(tibble())
  
  # --- 1. FAST SCAN ---
  observeEvent(input$run_scan, {
    tickers <- unlist(strsplit(input$manual_tickers, "\n"))
    
    if (!is.null(input$file_upload)) {
      ext <- tools::file_ext(input$file_upload$name)
      df_upload <- if(ext == "csv") read_csv(input$file_upload$datapath) else read.xlsx(input$file_upload$datapath)
      col <- names(df_upload)[str_which(tolower(names(df_upload)), "ticker|symbol")[1]]
      if (!is.na(col)) tickers <- c(tickers, df_upload[[col]])
    }
    
    tickers <- unique(trimws(toupper(tickers)))
    tickers <- tickers[tickers != ""]
    
    if(length(tickers) == 0) return()
    
    # Fast Loop
    withProgress(message = 'Scanning...', value = 0, {
      df <- map_dfr(tickers, function(t) {
        incProgress(1/length(tickers), detail = t)
        
        # Only get lightweight data
        price <- get_current_price(t)
        next_date <- get_next_earnings(t)
        
        tibble(
          Ticker = t,
          Price = price,
          `Next Earnings` = next_date,
          Action = paste0(
            '<button id="btn_', t, '" type="button" class="btn btn-info btn-sm" onclick="Shiny.setInputValue(\'select_ticker\', \'', t, '\', {priority: \'event\'})">🔍 Analyze</button>'
          )
        )
      })
    })
    
    summary_data(df)
  })
  
  # --- 2. RENDER MAIN TABLE ---
  output$summary_table <- renderDT({
    req(nrow(summary_data()) > 0)
    datatable(summary_data(), 
              escape = FALSE, # Allow HTML buttons
              selection = "none",
              rownames = FALSE,
              options = list(pageLength = 15, dom = 't')) %>%
      formatCurrency("Price", "$")
  })
  
  # --- 3. EXPAND / ANALYZE CLICK ---
  observeEvent(input$select_ticker, {
    ticker <- input$select_ticker
    
    showModal(modalDialog(
      title = paste("📊 Analysis for", ticker),
      size = "l", # Large modal
      
      # Loading spinner inside modal while data fetches
      renderUI({
        withProgress(message = 'Fetching History...', {
          hist_data <- get_history_analysis(ticker)
          
          if(is.null(hist_data)) {
            h4("No historical earnings data found.")
          } else {
            tagList(
              h5("Last 4 Quarters Reaction"),
              renderDT({
                 datatable(hist_data, rownames = FALSE, options = list(dom = 't')) %>%
                   formatPercentage(c("Surprise", "1D Move", "3D Move"), 2) %>%
                   formatStyle("1D Move", color = styleInterval(0, c('red', 'green')), fontWeight = 'bold')
              })
            )
          }
        })
      }),
      easyClose = TRUE,
      footer = modalButton("Close")
    ))
  })
}

shinyApp(ui, server)
