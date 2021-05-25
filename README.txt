        Author: Beckett Johnson


          Date: 3/13/2021


       Project: Collaborative Filtering Algorithms for Movie Recommendation


       Context: This project was assigned to me as a final project for my Web
                Information Management course wherein we explored recommendation
                system technologies in depth. This project in particular was meant to 
                mimic the well-known netflix competition which you can read more about 
                here: https://en.wikipedia.org/wiki/Netflix_Prize . In this project, I
		implement and discuss the following collaborative filtering algorithms
		for the application of movie recommendation:

			- User-based collaborative filtering, using cosine similarity 
			  for user similarity comparisons

			- User-based collaborative filtering, using pearson correlation
			  for user similarity comparisons

			- User-based collaborative filtering, using pearson correlation
			  for user similarity comparisons, adjusted with case modification

			- User-based collaborative filtering, using pearson correlation
			  for user similarity comparisons, adjusted by inverse user 
			  frequency

			- User-based collaborative filtering, using pearson correlation
			  for user similarity comparisons, adjusted by degree of movie 
			  rating polarity (my own algorithm design)

			- Item-based collaborative filtering, using cosine similarity 
			  for item similarity comparisons



      Contents: - "Project 2 Assignment Instructions.pdf" contains the prompt I was given for 
                   this assignment from my professor

                - "Beckett Johnson Project 2 Report.pdf" is the written report and explanation
                   that I wrote for this assignment which provides a high-level discussion of 
                   the technologies I employed and how they were implemented. It also discusses
                   the results from each test and provides analysis of the results.

                - "user_based_cosine.go" is a golang source file which contains my 
                   implementation of the user-based collaborative filtering algorithm using
                   cosine similarity for user similarity comparison.

                - "user_based_pearson.go" is a golang source file which contains my 
                   implementation of the user-based collaborative filtering algorithm using
                   pearson correlation for user similarity comparison.

                - "user_based_pearson(with_Case_Modification).go" is a golang source 
                   file which contains my implementation of the user-based collaborative
                   filtering algorithm using pearson correlation with case modification for 
                   user similarity comparison.

                - "user_based_pearson(with_Inverse_User_Frequency).go" is a golang source 
                   file which contains my implementation of the user-based collaborative
                   filtering algorithm using pearson correlation adjusted with Inverse User 
                   Frequency (IUF) for user similarity comparison.

                - "user_based_pearson(with_Polarizing_Movies_Emphasized).go" is a golang 
                   source file which contains my implementation of my own user-based collaborative
                   filtering algorithm variant which uses pearson correlation adjusted for rating
                   polarization for user similarity comparison.

                - "user_based_pearson(with_Case_Modification)(Real_Testing).go" is a golang 
                   source file which contains my implementation of the user-based collaborative
                   filtering algorithm using pearson correlation with case modification for 
                   user similarity comparison. This file was altered to be able to use additional
                   testing data (titled: 'test5.txt', 'test10.txt' & 'test20.txt') and to export it's 
                   predictions out into result files (titled: 'result5.txt', 'result10.txt' & 
                   'result20.txt').

                - "item_based_cosine.go" is a golang  source file which contains my implementation of
                   the user-based collaborative filtering algorithm using cosine similarity for item 
                   similarity comparison.

                - "train.txt" is a .txt file with all of the training data used by the programs listed above

                

