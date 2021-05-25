/*
Author: Beckett Johnson
Date: 3/13/2021
Description: This program implements the item-based cosine-similarity variant of collaborative 
			 filtering for the application of movie recommendation. This program uses data stored
			 in train.txt to implement and test the success of the collaborative filtering 
			 algorithm by using the first 900 movies as training data and the remaining 100 
			 movies' data for testing.
*/

package main

import (
	"bufio"
	"fmt"
	"io/ioutil"
	"math"
	"os"
	"strconv"
	"strings"
)

// Main function of program
func main() {
	// Retrieve data from training set and store it as a two dimensional array.
	// The first 900 movies' rating data is used as training data and the 
	// remaining 100 movies' ratings data is used as testing data.
	trainingData := getRatings()

	// Use training data to make predictions for testing data using cosine-similarity 
	// item-based collaborative filtering and then store those predictions
	predictions := makeAllPredictions(trainingData)

	// Find the RMSE for the testing data using the predicted values
	result := findRMSE(trainingData, predictions)

	// Print out the RMSE
	fmt.Printf("Item-Based Cosine Similarity RMSE: %f \n", result)
}

// Returns an [1000][200]int array with predictions made for all of the last 100 items' existing ratings
func makeAllPredictions(ratings [1000][200]int) [1000][200]int {
	var predictions [1000][200]int
	for row := 0; row < 200; row++ {
		for col := 900; col < 1000; col++ {
			if ratings[col][row] != 0 {
				predictions[col][row] = makeSinglePrediction(ratings, col, row)

				// if it predicts anything higher than 5
				if predictions[col][row] > 5 {
				//	fmt.Printf("you got a problem in making your prediction. \n")
				}
			}
		}
	}

	return predictions
}

// Returns a single prediction of what the active user would give the desired movie using ratings data from ratings[1000][200]
func makeSinglePrediction(ratings [1000][200]int, desiredMovie int, activeUser int) int {

	kSimilarMovieIndexes := [20]int{}              // Array with most 20 most similar movies
	kSimilarMovieSimilarityScores := [20]float64{} // Parallel array to 'kSimilarMoviesIndexes' that shows similarity scores
	leastSimilarIdx := 0                           // Index of the least similar movie within 'kSimilarMoviesIndexes'

	for otherMovie := 0; otherMovie < 900; otherMovie++ {
		if ratings[otherMovie][activeUser] != 0 {
			// Find the similarity score between 'otherMovie' and 'desiredMovie'
			currSimilarityScore := findMovieCosineSimilarity(ratings, desiredMovie, otherMovie)

			// If current otherMovie's similarity score is higher than the lowest of the previous 20 highest, then replace the least similar movie in 'kSimilarMovieIndexes'
			if currSimilarityScore > kSimilarMovieSimilarityScores[leastSimilarIdx] {
				kSimilarMovieIndexes[leastSimilarIdx] = otherMovie
				kSimilarMovieSimilarityScores[leastSimilarIdx] = currSimilarityScore
			}

			// Update 'leastSimilarIdx' with the index of the movie with the worst similarity score from 'kSimilarMoviesSimilarityScores'
			leastSimilarIdx = 0
			for idx := 1; idx < 20; idx++ {
				if kSimilarMovieSimilarityScores[idx] < kSimilarMovieSimilarityScores[leastSimilarIdx] {
					leastSimilarIdx = idx
				}
			}
		}
	}

	// We now have our list of the 20 most similar movies, time to compute prediction
	var sumOfSimilarityScores float64 = 0
	var sumOfSimilarityScoreTimesMovie2Rating float64 = 0
	var prediction float64

	// Calculate Sums
	for movie2 := 0; movie2 < 20; movie2++ {
		sumOfSimilarityScores += kSimilarMovieSimilarityScores[movie2]
		sumOfSimilarityScoreTimesMovie2Rating += (kSimilarMovieSimilarityScores[movie2] * float64(ratings[kSimilarMovieIndexes[movie2]][activeUser]))
	}

	prediction = sumOfSimilarityScoreTimesMovie2Rating / sumOfSimilarityScores

	//fmt.Printf("%f ~ %f ~ %d \n", prediction, math.Round(prediction), int(math.Round(prediction))) // for debugging purposes
	return int(math.Round(prediction))
}

// Returns the cosine similarity score between two movies, where movie1 and movie2 are indexes of movies in ratings[<movie_index>][200]
func findMovieCosineSimilarity(ratings [1000][200]int, movie1 int, movie2 int) float64 {
	var sumMovie1RatingsSqrd float64 = 0
	var sumMovie2RatingsSqrd float64 = 0
	var sumMovieRatingsMult float64 = 0

	for user := 0; user < 200; user++ {
		if (ratings[movie1][user] != 0) && (ratings[movie2][user] != 0) {

			var movie1Rating float64 = float64(ratings[movie1][user])
			var movie2Rating float64 = float64(ratings[movie2][user])

			sumMovie1RatingsSqrd += movie1Rating * movie1Rating

			sumMovie2RatingsSqrd += movie2Rating * movie2Rating

			sumMovieRatingsMult += movie1Rating * movie2Rating

		}
	}

	var similarity float64 = sumMovieRatingsMult / (math.Sqrt(sumMovie1RatingsSqrd) * math.Sqrt(sumMovie2RatingsSqrd))

	return similarity
}

// Uses the existing ratings from the last 100 movies to calculate findRMSE for the predicted ratings
func findRMSE(actual [1000][200]int, predicted [1000][200]int) float64 {
	noOfPredictedRatings := 0
	sumOfPredictedMinusActualSqrd := 0

	for row := 0; row < 200; row++ {
		for col := 900; col < 1000; col++ {
			if (predicted[col][row] != 0) && (predicted[col][row] != -9223372036854775808) {
				noOfPredictedRatings++
				sumOfPredictedMinusActualSqrd += ((predicted[col][row] - actual[col][row]) * (predicted[col][row] - actual[col][row]))
			}
		}
	}

	return math.Sqrt(float64(sumOfPredictedMinusActualSqrd) / float64(noOfPredictedRatings))
}

// getRatings retrieves data from training set and returns it as a two dimensional array
func getRatings() [1000][200]int {
	data, err := ioutil.ReadFile("train.txt")
	if err != nil {
		fmt.Println("File reading error", err)
		return [1000][200]int{}
	}

	scanner := bufio.NewScanner(strings.NewReader(string(data)))
	// Set the split function for the scanning words.
	scanner.Split(bufio.ScanWords)
	// store the words
	sample := [1000][200]int{}

	for row := 0; row < 200; row++ {
		for col := 0; col < 1000; col++ {
			scanner.Scan()
			currStr, _ := strconv.Atoi(scanner.Text())
			sample[col][row] = currStr
		}
	}

	if err := scanner.Err(); err != nil {
		fmt.Fprintln(os.Stderr, "reading input:", err)
	}

	return sample
}

// prints out the values in an 1000 by 200 integer array; used for debugging
func printArray200(sample [1000][200]int) {

	for row := 0; row < 200; row++ {
		for col := 0; col < 1000; col++ {
			fmt.Printf("%d ", sample[col][row])
		}
		fmt.Printf("\n")
	}
}
