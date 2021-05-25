/*
Author: Beckett Johnson
Date: 3/13/2021
Description: This program implements my own user-based pearson-correlation variant of collaborative
			 filtering for the application of movie recommendation. As outlined in more detail in
			 my project report, my collaborative filtering variant is based upon the intuition that
			 a movie which has polarized ratings (i.e. has a comparatively large standard deviation),
			 will be more indicative of a user's taste in movies and therefore should be given more
			 wieghting in a prediction. This program uses data stored in train.txt to implement and
			 test the success of the collaborative filtering algorithm by using the first 175 users
			 as training data and the remaining 25 users' data for testing.
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
	// Retrieve data from training set and store it as a two dimensional array
	// The first 175 user's data is used as training data and the remaining 25 user's data is used as testing data
	trainingData := getRatings()

	// Takes the training data for all the users and returns new ratings which are adjusted using an 'polarization' multiplier
	// These adjusted ratings are only used when trying to find correlated users to base predictions off of
	adjustedTrainingData := emphasizeControversialMovies(trainingData)

	// Use training data to make predictions for testing data using pearson-correlation user-based collaborative filtering with inverse user frequency
	// Then store these predictions
	predictions := makeAllAdjustedPredictions(trainingData, adjustedTrainingData)

	// Find the RMSE for the testing data using the predicted values
	result := findRMSE(trainingData, predictions)

	// Print out the RMSE
	fmt.Printf("User-Based Pearson Correlation with Polarizing Movies Emphasized RMSE: %f \n", result)
}

// Takes the training data for all the users and returns new ratings which are adjusted using an 'polarization' multiplier
// These adjusted ratings are only used when trying to find correlated users to base predictions off of
func emphasizeControversialMovies(ratings [1000][200]int) [1000][200]float64 {
	var adjustedRatings [1000][200]float64
	var movieSDs [1000]float64 // Holds the standard deviations for all movies

	// Store the standard deviation for each movie in 'movieSDs'
	for movie := 0; movie < 1000; movie++ {
		noOfRatings := 0
		sumOfRatings := 0

		for user := 0; user < 200; user++ {
			if ratings[movie][user] != 0 {
				noOfRatings++
				sumOfRatings += ratings[movie][user]
			}
		}

		if noOfRatings == 1 {
			movieSDs[movie] = 0
		} else {
			var avgRating float64 = float64(sumOfRatings) / float64(noOfRatings)
			var sumForSD float64 = 0

			for user := 0; user < 200; user++ {
				if ratings[movie][user] != 0 {
					sumForSD += math.Pow((float64(ratings[movie][user]) - avgRating), 2)
				}
			}

			movieSDs[movie] = math.Sqrt(sumForSD / float64(noOfRatings-1))
		}
	}

	// Find average standard deviation for all movies in the dataset
	var avgStandardDeviation float64 = 0

	for movie := 0; movie < 1000; movie++ {
		avgStandardDeviation += movieSDs[movie]
	}

	avgStandardDeviation = avgStandardDeviation / 1000

	for movie := 0; movie < 1000; movie++ {
		for user := 0; user < 200; user++ {
			if ratings[movie][user] != 0 {
				adjustedRatings[movie][user] = float64(ratings[movie][user]) * (math.Logb(movieSDs[movie]) - math.Logb(avgStandardDeviation))
			}
		}
	}

	return adjustedRatings
}

// Returns an [1000][200]int array with predictions made for all of the last 25 user's existing ratings
func makeAllAdjustedPredictions(ratings [1000][200]int, adjustedRatings [1000][200]float64) [1000][200]int {
	var predictions [1000][200]int

	for row := 175; row < 200; row++ {

		for col := 0; col < 1000; col++ {
			if ratings[col][row] != 0 {
				predictions[col][row] = makeSingleAdjustedPrediction(ratings, adjustedRatings, col, row)

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
func makeSingleAdjustedPrediction(ratings [1000][200]int, adjustedRatings [1000][200]float64, desiredMovie int, activeUser int) int {

	kSimilarUsersIndexes := [20]int{}              // Array with most 20 most similar users
	kSimilarUsersSimilarityScores := [20]float64{} // Parallel array to 'kSimilarUsersIndexes' that shows similarity scores
	leastSimilarIdx := 0                           // Index of the least similar user within 'kSimilarUsersIndexes'

	activeUserAvgRating := findUserAverageRatingWithFloats(convIntToFloat(ratings), activeUser)

	for otherUser := 0; otherUser < 175; otherUser++ {
		if ratings[desiredMovie][otherUser] != 0 {
			// Find the similarity score between 'otherUser' and 'activeUser'
			currSimilarityScore := findAdjustedUserPearsonSimilarity(ratings, adjustedRatings, activeUser, otherUser)

			// If current user's similarity score is higher the the previous 20 highest, then replace the least similar user in 'kSimilarUsersIndexes'
			if math.Abs(currSimilarityScore) > math.Abs(kSimilarUsersSimilarityScores[leastSimilarIdx]) {
				kSimilarUsersIndexes[leastSimilarIdx] = otherUser
				kSimilarUsersSimilarityScores[leastSimilarIdx] = currSimilarityScore
			}

			// Update 'leastSimilarIdx' with the index of the user with the worst similarity score from 'kSimilarUsersSimilarityScores'
			leastSimilarIdx = 0
			for idx := 1; idx < 20; idx++ {
				if math.Abs(kSimilarUsersSimilarityScores[idx]) < math.Abs(kSimilarUsersSimilarityScores[leastSimilarIdx]) {
					leastSimilarIdx = idx
				}
			}
		}
	}

	// We now have our list of the 20 most similar users, time to compute prediction
	var summation1 float64 = 0 // Represents: summation(Similarity_Score * (User_2_Movie_Rating - User_2_Avg_Rating))
	var summation2 float64 = 0 // Represents: summation( abs(Similarity_Score) )

	for user2 := 0; user2 < 20; user2++ {
		summation1 += kSimilarUsersSimilarityScores[user2] * (float64(ratings[desiredMovie][kSimilarUsersIndexes[user2]]) - findUserAverageRatingWithFloats(convIntToFloat(ratings), kSimilarUsersIndexes[user2]))
		summation2 += math.Abs(kSimilarUsersSimilarityScores[user2])
	}

	prediction := activeUserAvgRating + (summation1 / summation2)

	if math.IsNaN(prediction) {
		prediction = activeUserAvgRating
	} else if (prediction == 0) || (prediction == -1) {
		prediction = 1
	} else if (prediction == 6) || (prediction == 7) {
		prediction = 5
	}

	//fmt.Printf("%f ~ %f ~ %d \n", prediction, math.Round(prediction), int(math.Round(prediction))) // for debugging purposes
	return int(math.Round(prediction))
}

// Returns the cosine similarity score between two users, where user1 and user2 are indexes of users in ratings[1000][<user_index>]
func findAdjustedUserPearsonSimilarity(ratings [1000][200]int, adjustedRatings [1000][200]float64, activeUser int, user2 int) float64 {
	var summation1 float64 = 0 // Represents: summation( (Active_User_Movie_Rating - Active_User_Avg_Rating) * (User_2_Movie_Rating - User_2_Avg_Rating) )
	var summation2 float64 = 0 // Represents: summation( squared(Active_User_Movie_Rating - Active_User_Avg_Rating) )
	var summation3 float64 = 0 // Represents: summation( squared(User_2_Movie_Rating - User_2_Avg_Rating) )

	activeUserAvgRating := findUserAverageRatingWithFloats(adjustedRatings, activeUser)
	user2AvgRating := findUserAverageRatingWithFloats(adjustedRatings, user2)

	for movie := 0; movie < 1000; movie++ {
		if (ratings[movie][activeUser] != 0) && (ratings[movie][user2] != 0) {

			var normalizedActiveUserRating float64 = (float64(adjustedRatings[movie][activeUser]) - activeUserAvgRating)
			var normalizedUser2Rating float64 = (float64(adjustedRatings[movie][user2]) - user2AvgRating)

			summation1 += normalizedActiveUserRating * normalizedUser2Rating

			summation2 += normalizedActiveUserRating * normalizedActiveUserRating

			summation3 += normalizedUser2Rating * normalizedUser2Rating
		}
	}

	if summation3 == 0 {
		return 0
	}

	var similarity float64 = summation1 / (math.Sqrt(summation2) * math.Sqrt(summation3))

	return similarity
}

func convIntToFloat(ratings [1000][200]int) [1000][200]float64 {
	new := [1000][200]float64{}

	for row := 0; row < 200; row++ {
		for col := 0; col < 1000; col++ {
			new[col][row] = float64(ratings[col][row])
		}
	}

	return new
}

// Returns a float64 that represents a user's average rating
func findUserAverageRatingWithFloats(ratings [1000][200]float64, user int) float64 {
	var sumOfRatings float64 = 0
	var noOfRatings int

	for movie := 0; movie < 1000; movie++ {
		if ratings[movie][user] != 0 {
			noOfRatings++
			sumOfRatings += float64(ratings[movie][user])
		}
	}

	return sumOfRatings / float64(noOfRatings)
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
func printIntArray200(sample [1000][200]int) {

	for row := 0; row < 200; row++ {
		for col := 0; col < 1000; col++ {
			fmt.Printf("%d ", sample[col][row])
		}
		fmt.Printf("\n")
	}
}

// prints out the values in an 1000 by 200 float64 array; used for debugging
func printFloat64Array200(sample [1000][200]float64) {

	for row := 0; row < 200; row++ {
		for col := 0; col < 1000; col++ {
			fmt.Printf("%f ", sample[col][row])
		}
		fmt.Printf("\n")
	}
}
