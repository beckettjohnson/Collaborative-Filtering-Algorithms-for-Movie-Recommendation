/*
Author: Beckett Johnson
Date: 3/13/2021
Description: This program implements the user-based cosine-similarity variant of collaborative filtering
			 for the application of movie recommendation. This program uses data stored in
			 train.txt to implement and test the success of the collaborative filtering
			 algorithm by using the first 175 users as training data and the remaining 25
			 users' data for testing.
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

// Main function
func main() {

	// Retrieve data from training set and store it as a two dimensional array
	// The first 175 user's data is used as training data and the remaining 25 user's data is used as testing data
	trainingData := getRatings()

	// Use training data to make predictions for testing data using cosine-similarity user-based collaborative filtering
	// Then store these predictions
	predictions := makeAllPredictions(trainingData)

	// Find the RMSE for the testing data using the predicted values
	result := findRMSE(trainingData, predictions)

	// Print out the RMSE
	fmt.Printf("User-Based Cosine Similarity RMSE: %f \n", result)
}

// Returns an [1000][200]int array with predictions made for all of the last 25 user's existing ratings
func makeAllPredictions(ratings [1000][200]int) [1000][200]int {
	var predictions [1000][200]int
	for row := 175; row < 200; row++ {
		for col := 0; col < 1000; col++ {
			if ratings[col][row] != 0 {
				predictions[col][row] = makeSinglePrediction(ratings, col, row)

				// if it predicts anything higher than 5
				if predictions[col][row] > 5 {
					fmt.Printf("you got a problem in making your prediction. \n")
				}
			}
		}
	}

	return predictions
}

// Returns a single prediction of what the desired user would give the desired movie using ratings data from ratings[1000][200]
func makeSinglePrediction(ratings [1000][200]int, desiredMovie int, activeUser int) int {

	kSimilarUsersIndexes := [20]int{}              // Array with most 20 most similar users
	kSimilarUsersSimilarityScores := [20]float64{} // Parallel array to 'kSimilarUsersIndexes' that shows similarity scores
	leastSimilarIdx := 0                           // Index of the least similar user within 'kSimilarUsersIndexes'

	for otherUser := 0; otherUser < 175; otherUser++ {
		if ratings[desiredMovie][otherUser] != 0 {
			// Find the similarity score between 'otherUser' and 'activeUser'
			currSimilarityScore := findUserCosineSimilarity(ratings, activeUser, otherUser)

			// If current user's similarity score is higher the the previous 20 highest, then replace the least similar user in 'kSimilarUsersIndexes'
			if currSimilarityScore > kSimilarUsersSimilarityScores[leastSimilarIdx] {
				kSimilarUsersIndexes[leastSimilarIdx] = otherUser
				kSimilarUsersSimilarityScores[leastSimilarIdx] = currSimilarityScore
			}

			// Update 'leastSimilarIdx' with the index of the user with the worst similarity score from 'kSimilarUsersSimilarityScores'
			leastSimilarIdx = 0
			for idx := 1; idx < 20; idx++ {
				if kSimilarUsersSimilarityScores[idx] < kSimilarUsersSimilarityScores[leastSimilarIdx] {
					leastSimilarIdx = idx
				}
			}
		}
	}

	// We now have our list of the 20 most similar users, time to compute prediction
	var sumOfSimilarityScores float64 = 0
	var sumOfSimilarityScoreTimesUser2Rating float64 = 0
	var prediction float64

	for user2 := 0; user2 < 20; user2++ {
		sumOfSimilarityScores += kSimilarUsersSimilarityScores[user2]
		sumOfSimilarityScoreTimesUser2Rating += (kSimilarUsersSimilarityScores[user2] * float64(ratings[desiredMovie][kSimilarUsersIndexes[user2]]))
	}

	prediction = sumOfSimilarityScoreTimesUser2Rating / sumOfSimilarityScores

	if (prediction == 0) || (prediction == -1) {
		prediction = 1
	} else if (prediction == 6) || (prediction == 7) {
		prediction = 5
	}

	//fmt.Printf("%f ~ %f ~ %d \n", prediction, math.Round(prediction), int(math.Round(prediction))) // for debugging purposes
	return int(math.Round(prediction))
}

// Returns the cosine similarity score between two users, where user1 and user2 are indexes of users in ratings[1000][<user_index>]
func findUserCosineSimilarity(ratings [1000][200]int, user1 int, user2 int) float64 {
	var sumUser1RatingsSqrd float64 = 0
	var sumUser2RatingsSqrd float64 = 0
	var sumOfUserMovieRatingsMult float64 = 0

	for movie := 0; movie < 1000; movie++ {
		if (ratings[movie][user1] != 0) && (ratings[movie][user2] != 0) {

			var user1rating float64 = float64(ratings[movie][user1])
			var user2rating float64 = float64(ratings[movie][user2])

			sumUser1RatingsSqrd = sumUser1RatingsSqrd + (user1rating * user1rating)

			sumUser2RatingsSqrd = sumUser2RatingsSqrd + (user2rating * user2rating)

			sumOfUserMovieRatingsMult = sumOfUserMovieRatingsMult + (user1rating * user2rating)

		}
	}

	var similarity float64 = sumOfUserMovieRatingsMult / (math.Sqrt(sumUser1RatingsSqrd) * math.Sqrt(sumUser2RatingsSqrd))

	return similarity
}

// Uses the existing ratings from the last 25 users to calculate findRMSE for the predicted ratings
func findRMSE(actual [1000][200]int, predicted [1000][200]int) float64 {
	noOfPredictedRatings := 0
	sumOfPredictedMinusActualSqrd := 0

	for row := 175; row < 200; row++ {
		for col := 0; col < 1000; col++ {
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
