/*
Author: Beckett Johnson
Date: 3/13/2021
Description: This program implements the user-based pearson-correlation (with case modification) variant of collaborative filtering
			 for the application of movie recommendation. This program uses all of the data stored in
			 train.txt as training data to predict user ratings for files test5.txt, test10.txt &
			 test20.txt; storing results in files result5.txt, result10.txt & result20.txt .
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
	trainingData := getTrainRatings()

	// Make predictions with training data for testing data in test5.txt, then store it in result5.txt
	testingData := getTestRatings("test5.txt", trainingData)
	predictions := makeAllPredictions(testingData)
	exportResults(testingData, predictions, "test5.txt", "result5.txt")

	// Make predictions with training data for testing data in test10.txt, then store it in result10.txt
	testingData = getTestRatings("test10.txt", trainingData)
	predictions = makeAllPredictions(testingData)
	exportResults(testingData, predictions, "test10.txt", "result10.txt")

	// Make predictions with training data for testing data in test20.txt, then store it in result20.txt
	testingData = getTestRatings("test20.txt", trainingData)
	predictions = makeAllPredictions(testingData)
	exportResults(testingData, predictions, "test20.txt", "result20.txt")
}

// Returns an [1000][200]int array with predictions made for all of the last 25 user's existing ratings
func makeAllPredictions(ratings [1000][300]int) [1000][300]int {
	var predictions [1000][300]int = ratings

	for row := 200; row < 300; row++ {

		for col := 0; col < 1000; col++ {
			if ratings[col][row] == -1 {
				predictions[col][row] = makeSinglePrediction(ratings, col, row)

				/*// if it predicts anything higher than 5
				if predictions[col][row] > 5 {
					fmt.Printf("you got a problem in making your prediction. \n")
				}
				*/
			}
		}
	}
	return predictions
}

// Returns a single prediction of what the desired user would give the desired movie using ratings data from ratings[1000][200]
func makeSinglePrediction(ratings [1000][300]int, desiredMovie int, activeUser int) int {

	kSimilarUsersIndexes := [20]int{}              // Array with most 20 most similar users
	kSimilarUsersSimilarityScores := [20]float64{} // Parallel array to 'kSimilarUsersIndexes' that shows similarity scores
	leastSimilarIdx := 0                           // Index of the least similar user within 'kSimilarUsersIndexes'

	activeUserAvgRating := findUserAverageRating(ratings, activeUser)

	for otherUser := 0; otherUser < 200; otherUser++ {
		if ratings[desiredMovie][otherUser] != 0 {
			// Find the similarity score between 'otherUser' and 'activeUser'
			currSimilarityScore := findUserPearsonSimilarity(ratings, activeUser, otherUser, activeUserAvgRating)

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
		summation1 += kSimilarUsersSimilarityScores[user2] * (float64(ratings[desiredMovie][kSimilarUsersIndexes[user2]]) - findUserAverageRating(ratings, kSimilarUsersIndexes[user2]))
		summation2 += math.Abs(kSimilarUsersSimilarityScores[user2])
	}

	prediction := activeUserAvgRating + (summation1 / summation2)

	if math.IsNaN(prediction) {
		prediction = activeUserAvgRating
	}

	//fmt.Printf("%f ~ %f ~ %d \n", prediction, math.Round(prediction), int(math.Round(prediction))) // for debugging purposes
	return int(math.Round(prediction))
}

// Returns the cosine similarity score between two users, where user1 and user2 are indexes of users in ratings[1000][<user_index>]
func findUserPearsonSimilarity(ratings [1000][300]int, activeUser int, user2 int, activeUserAvgRating float64) float64 {
	var summation1 float64 = 0 // Represents: summation( (Active_User_Movie_Rating - Active_User_Avg_Rating) * (User_2_Movie_Rating - User_2_Avg_Rating) )
	var summation2 float64 = 0 // Represents: summation( squared(Active_User_Movie_Rating - Active_User_Avg_Rating) )
	var summation3 float64 = 0 // Represents: summation( squared(User_2_Movie_Rating - User_2_Avg_Rating) )

	user2AvgRating := findUserAverageRating(ratings, user2)

	for movie := 0; movie < 1000; movie++ {
		if (ratings[movie][activeUser] != 0) && (ratings[movie][activeUser] != -1) && (ratings[movie][user2] != 0) {

			var normalizedActiveUserRating float64 = (float64(ratings[movie][activeUser]) - activeUserAvgRating)
			var normalizedUser2Rating float64 = (float64(ratings[movie][user2]) - user2AvgRating)

			summation1 += normalizedActiveUserRating * normalizedUser2Rating

			summation2 += normalizedActiveUserRating * normalizedActiveUserRating

			summation3 += normalizedUser2Rating * normalizedUser2Rating
		}
	}

	if summation3 == 0 {
		return 0
	}

	var similarity float64 = summation1 / (math.Sqrt(summation2) * math.Sqrt(summation3))

	p := 1.0

	caseModSimilarity := similarity * math.Abs(math.Pow(similarity, p-1))

	if math.IsNaN(caseModSimilarity) {
		caseModSimilarity = 0
	}

	return caseModSimilarity
}

// Returns a float64 that represents a user's average rating
func findUserAverageRating(ratings [1000][300]int, user int) float64 {
	var sumOfRatings float64 = 0
	var noOfRatings int

	for movie := 0; movie < 1000; movie++ {
		if (ratings[movie][user] != 0) && (ratings[movie][user] != -1) {
			noOfRatings++
			sumOfRatings += float64(ratings[movie][user])
		}
	}

	return sumOfRatings / float64(noOfRatings)
}

// This function writes the predicted values into the appropriate result file
func exportResults(original [1000][300]int, predicted [1000][300]int, inputFileName string, outputFileName string) {
	var userIDConstant int

	if inputFileName == "test5.txt" {
		userIDConstant = 0
	} else if inputFileName == "test10.txt" {
		userIDConstant = 100
	} else if inputFileName == "test20.txt" {
		userIDConstant = 200
	}

	outputFileName = "./" + outputFileName

	file, err := os.Create(outputFileName)
	if err != nil {
		fmt.Println("File creating error", err)
	}
	writer := bufio.NewWriter(file)

	linesToWrite := []string{}

	var currentLine string = ""

	for user := 200; user < 300; user++ {
		for movie := 0; movie < 1000; movie++ {
			if original[movie][user] == -1 {

				nextRating := predicted[movie][user]

				if (nextRating == 0) || (nextRating == -1) {
					nextRating = 1
				} else if (nextRating == 6) || (nextRating == 7) {
					nextRating = 5
				}

				currentLine = strconv.Itoa(user+userIDConstant+1) + " " + strconv.Itoa(movie+1) + " " + strconv.Itoa(nextRating) + "\n"
				linesToWrite = append(linesToWrite, currentLine)
			}
		}
	}

	for _, line := range linesToWrite {
		_, err := writer.WriteString(line + "\n")
		if err != nil {
			fmt.Println("Writing to file error", err)
		}
	}

	writer.Flush()

}

// getRatings retrieves data from training set and returns it as a two dimensional array
func getTrainRatings() [1000][300]int {
	data, err := ioutil.ReadFile("train.txt")
	if err != nil {
		fmt.Println("File reading error", err)
		return [1000][300]int{}
	}

	scanner := bufio.NewScanner(strings.NewReader(string(data)))
	// Set the split function for the scanning words.
	scanner.Split(bufio.ScanWords)
	// store the words
	sample := [1000][300]int{}

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

// GetTestRatings retrieves data from training set and returns it as a two dimensional array
func getTestRatings(filename string, ratings [1000][300]int) [1000][300]int {
	data, err := ioutil.ReadFile(filename)
	if err != nil {
		fmt.Println("File reading error", err)
		return [1000][300]int{}
	}

	scanner := bufio.NewScanner(strings.NewReader(string(data)))
	// Set the split function for the scanning words.
	scanner.Split(bufio.ScanWords)
	// store the words
	sample := ratings

	processNo := 0

	var userIdx int
	var movieIdx int

	for scanner.Scan() {
		if processNo == 0 {
			currStr, _ := strconv.Atoi(scanner.Text())
			userIdx = ((currStr - 1) % 100) + 200
			//fmt.Println("%d ", userIdx)
		} else if processNo == 1 {
			currStr, _ := strconv.Atoi(scanner.Text())
			movieIdx = currStr - 1
			//fmt.Println("%d ", movieIdx)
		} else {
			currStr, _ := strconv.Atoi(scanner.Text())
			if currStr == 0 {
				sample[movieIdx][userIdx] = -1
			} else {
				sample[movieIdx][userIdx] = currStr
			}
			//fmt.Println("%d\n", sample[movieIdx][userIdx])
		}

		processNo = (processNo + 1) % 3
	}

	if err := scanner.Err(); err != nil {
		fmt.Fprintln(os.Stderr, "reading input:", err)
	}

	return sample
}

// prints out the values in an 1000 by 200 integer array; used for debugging
func printArray300(sample [1000][300]int) {

	for row := 0; row < 300; row++ {
		for col := 0; col < 1000; col++ {
			fmt.Printf("%d ", sample[col][row])
		}
		fmt.Printf("\n")
	}
}
