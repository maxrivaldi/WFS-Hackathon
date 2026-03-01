package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/xitongsys/parquet-go-source/local"
	"github.com/xitongsys/parquet-go/writer"
)

type FinnhubNews struct {
	Datetime int64  `json:"datetime"`
	Headline string `json:"headline"`
	Source   string `json:"source"`
}

type NewsRecord struct {
	Title  string `parquet:"name=title, type=BYTE_ARRAY, convertedtype=UTF8, encoding=PLAIN_DICTIONARY"`
	Source string `parquet:"name=source, type=BYTE_ARRAY, convertedtype=UTF8, encoding=PLAIN_DICTIONARY"`
	Date   string `parquet:"name=date, type=BYTE_ARRAY, convertedtype=UTF8, encoding=PLAIN_DICTIONARY"`
}

func main() {
	// 1. Define flags for input
	startDatePtr := flag.String("start", "Na", "Start date in YYYY-MM-DD format")
	daysPtr := flag.Int("days", 0, "Number of days to look back")

	tickerPtr := flag.String("ticker", "", "Stock Ticker")
	stockPtr := flag.String("stock", "", "Stock Name")

	flag.Parse()

	apiKey := "d6hkh6hr01qr5k4cb93gd6hkh6hr01qr5k4cb940"

	// 2. Parse the input start date
	baseDate, err := time.Parse("2006-01-02", *startDatePtr)
	if err != nil {
		log.Fatalf("Invalid date format. Use YYYY-MM-DD: %v", err)
	}

	fw, err := local.NewLocalFileWriter("mentions.parquet")
	if err != nil {
		log.Fatal(err)
	}
	pw, err := writer.NewParquetWriter(fw, new(NewsRecord), 4)
	if err != nil {
		log.Fatal(err)
	}

	for i := 0; i < *daysPtr; i++ {
		// 3. Subtract days from the PROVIDED start date instead of 'Now'
		currentDate := baseDate.AddDate(0, 0, -i)
		dateStr := currentDate.Format("2006-01-02")

		fmt.Printf("Checking %s... ", dateStr)

		// Updated symbol parameter to 'ticker'
		url := fmt.Sprintf(
			"https://finnhub.io/api/v1/company-news?symbol=%s&from=%s&to=%s&token=%s",
			*tickerPtr, dateStr, dateStr, apiKey,
		)

		resp, err := http.Get(url)
		if err != nil {
			fmt.Printf("Network Error: %v\n", err)
			continue
		}

		var news []FinnhubNews
		if err := json.NewDecoder(resp.Body).Decode(&news); err != nil {
			resp.Body.Close()
			fmt.Printf("Decode Error\n")
			continue
		}
		resp.Body.Close()

		matchCount := 0
		for _, article := range news {
			h := strings.ToLower(article.Headline)

			// Dynamically check for the stock name or ticker passed in via flags
			if strings.Contains(h, strings.ToLower(*stockPtr)) || strings.Contains(h, strings.ToLower(*tickerPtr)) {
				record := NewsRecord{
					Title:  article.Headline,
					Source: article.Source,
					Date:   dateStr,
				}
				pw.Write(record)
				matchCount++
			}
		}

		if matchCount == 0 {
			fmt.Print("No matches. Writing placeholder.")
			pw.Write(NewsRecord{Title: "", Source: "NONE", Date: dateStr})
		} else {
			fmt.Printf("Found %d matches.", matchCount)
		}
		fmt.Println()
		time.Sleep(1 * time.Second)
	}

	pw.WriteStop()
	fw.Close()
	fmt.Println("\nFinished! Check mentions.parquet")
}
