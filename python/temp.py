# % --------------------------- Bulls and Cows -------------------------- % #

# num_1 = "1234" # Todo Choose randomly

# MAX_ATTEMPT = 3
# attempt_number = 0

# while attempt_number != MAX_ATTEMPT:
#     attempt_number += 1
#     print(f"Փորձ #{attempt_number}")
    
#     num_2 = input("Please enter your guess: ")
#     # check num_2 is 4 digit long. 
#     # contains only digits
#     # no duplication    
#     count_cows = 0
#     count_bulls = 0
#     for i in range(len(num_1)):
#         # ------ Cows  -------
#         if num_1[i] in num_2:
#             count_cows += 1

#         # ------ Bulls -------
#         if num_1[i] == num_2[i]:
#             count_bulls += 1

#     if count_bulls == len(num_1):
#         print("Հաղթանական")
#         break
#     else:
#         print(count_cows, count_bulls)
        
# else:
#     print("Պարտվել էս")

# % --------------------------- Hangman -------------------------- % #
# 1. Choose random word
# 2. Get letter
# 3. Check letter in word (run a loop)
# 4. Check won
# 5. Check lost

import random

MAX_ATTEMPTS = 5
num_mistakes = 0

words = ["cheese", "elephant", "chocolate", "ban"]

word = random.choice(words)

guess = ["_"] * len(word)

letter_history = []

while num_mistakes < MAX_ATTEMPTS:
    print(guess)
    print(f"Օգտագործված տառեր {letter_history}")
    print("❤️"*(MAX_ATTEMPTS - num_mistakes))
    letter = input("Input a letter: ")
    
    if len(letter) != 1:
        print("Please only input one letter")
        continue
        
    if letter in letter_history:
        print(f"Արդեն փորձել էս {letter} տառը")
        continue
    
    letter_history.append(letter)
    
    # if letter not in word:
    #     num_mistakes += 1
    #     continue
    
    gtac_tar = False
    
    for i in range(len(word)):
        if letter == word[i]:
            gtac_tar = True
            guess[i] = letter

    if not gtac_tar:
        num_mistakes += 1 
        continue
    
    if '_' not in guess:
        print("Հաղթանակ")
        break
            


    
    