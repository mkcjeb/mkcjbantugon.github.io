---
title: "Dead or Alive: The Game"
date: 2019-10-14
tags: [python, user-input, game development, text adventure]
header:
  image: "/images/The-walking-dead.jpg"
excerpt: "(Python) A text input interactive game full of random plot twists and survival strategy.This game requires courage, and having a small amount of survival knowledge in case of a zombie apocalypse will be helpful. Desgined in Jupyter Notebooks."
mathjax: "true"
---

## The Game

Fan of intense and survival games? Rejoice! <br>
Let's refresh your memory on the game's inspiration drawn from the worlds of <a href="https://www.sportskeeda.com/pop-culture/what-is-alice-in-borderland-about">Alice in Borderland</a> and <a href="https://movieweb.com/escape-room-ending-explained/">Escape Room</a> movies.
    
In the realm of <b>Alice in Borderland</b>, characters find themselves in a distant fantasy land called Borderland, where survival hinges on engaging in death-defying challenges to earn precious lives. Win, and you gain additional days to live; lose, and it could cost you your life. When their visas expire, players face the threat of being instantly eliminated by a laser. The only way to stay alive is to engage in a series of challenging games.

In another world, <b>Escape Room</b>, six strangers receive mysterious black boxes with tickets to an immersive escape room, promising a chance to win a big amount of money. Trapped in various rooms with extreme challenges, players unravel the secrets within to survive and secure their escape.

And now, the game is set for you. Your challenge?

As the player, you will find yourself trapped in a room, navigating through a sequence of trials and questions. Your survival depends on relying on your wit and creative problem-solving skills to persevere and crack the challenges that lie ahead. 

Anticipate the unexpected in this game, comprising <u><font color=blue>three stages</u></font> featuring some unforeseen twists distinct from those depicted in the movie-adapted version. You will face different challenges with <b>Kings of</b> <b><font color=red>Hearts, Diamonds</b>,</font> or <b>Spades</b>. It involves a mix of chance and strategy, testing the player's decision-making skills and wit. Players progress through challenges, face unique scenarios, and ultimately aim to escape the Python Escape Room.

Best of luck on your journey!

## Instructions

You must have Python installed in your computer in order to play the game. Open your terminal window and run the jupyter notebook and copy-paste the following code block. Then, you only need to run the cell to start the game and follow the printed commands!

*Note: If you're not familiar with Jupyter Notebooks, here's a [great tutorial](https://www.youtube.com/watch?v=HW29067qVWk) on how they work and [how to set up](https://www.anaconda.com/products/individual) Anaconda Navigator.*


```python
"""
Created on Mon December 03, 2023

@author: michelle.bantugon
"""

"""
# Importing modules for generating random values, strings, and time
import random
import string
import time

#####################################################################################
# Defining a game_intro function to introduce the game and prompt the player to start
#####################################################################################

def game_intro():
    # Displaying a welcome message
    print("\nHey there, Player!🎉 Ready to enter into the wild world of Python Escape Room?🤑🤑🤑\n")
    
    # Calling the name function to get the player's username
    username = name() 
    
    # Displaying the obtained username
    print(f"\nAre you ready, {username.title()}? You're at the point of no return!🚷")
    
    input(prompt = "Press <<Enter>> to start.\n")
    
    # Calling the stage_1 function to start the first stage of the game
    stage_1() 

# Defining a name function to prompt the user to input their name or a random username will be generated
def name():
    
    # Using a loop to repeatedly ask for a valid username
    while True:
        
        # Getting user input for the username
        username = input(prompt = "Input your name: ")
        
        # Checking if the username input is not empty
        if username.strip():
            
            # Checking if the username input contains only letters or spaces
            if all(x.isalpha() or x.isspace() for x in username):
                # Returning the valid username
                return username 
            
            # Displaying a message for invalid input
            else:
                print("Invalid input. Please enter a valid string without numbers or special characters.\n")
        
        # Generating a random username if the input is empty
        else:
            return random_username()

# Defining a random_username function to generate a random username
def random_username():
    
    # Generating a random username using letters and digits with a length of 6 characters
    random_username = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
    
    # Displaying a message with the generated random username
    print(f"Oh, feeling nervous! You will be called... something mysterious🕵️‍♂️: {random_username}")

    # Returning the generated random username
    return random_username

#####################################################################################
# Defining a stage_1 function for the first stage of the game
#####################################################################################

def stage_1():
    
    # Displaying an introduction message for Stage 1
    print(f"{'*' * 120}")
    print("""Stage 1️⃣: Meeting the Kings🚪♥️👑♦️👑♠️\n
    You stand in a room with three doors: left, middle, and right.🤔
    """)
    
    # Using a loop to handle player's choice for door
    while True:
        
        # Getting user input for the chosen door
        action1 = input(prompt = "Which way do you want to go? ╰┈➤🚪").casefold()
        
        # Meeting the King of Hearts if chosen door is left
        if "left" in action1 and "middle" not in action1 and "right" not in action1:
            
            # Displaying a message for meeting the King of Hearts
            print(f"\n{' ♥️' * 60}\n""")
            print("Welcome, left-handed adventurer!🧙 The King of Hearts eagerly awaits your arrival in his kingdom.❤️‍🔥♖\n")
            input(prompt = "Press <<Enter>> to continue.\n")
            print("""King of Hearts: I challenge you to a single round of the ultimate game of Rock, Paper, Scissors.🪨📰✂
May the odds be in your favor!👸💀\n
            """)
            
            # Prompting the player to press Enter 
            input(prompt = "Press <<Enter>> to start the game.\n")

            # Initializing variables needed for Rock, Paper, Scissor game
            option1 = ["Rock", "Paper", "Scissor"]
            player_score_1 = 0
            king_hearts_score = 0

            # Using a loop to handle game for only one round
            while player_score_1 < 1 and king_hearts_score < 1:
                
                # Getting user input for the player's choice
                choice1 = input(prompt = "Choose Rock, Paper, or Scissor: ").capitalize()

                # Checking if the player's choice is valid
                if any(option.lower() in choice1.lower() for option in option1) and \
                    sum(option.lower() in choice1.lower() for option in option1) == 1:
                    choice1 = next(option for option in option1 if option.lower() in choice1.lower())
                    
                    # Assigning a random choice for the King of Hearts
                    king_choice = random.choice(option1)
                    
                    # Displaying both the player's and King's choices
                    print("You chose:", choice1)
                    print("King of Hearts chose:", king_choice)

                    # No one wins
                    if choice1 == king_choice:
                        print("\nIt's a tie!😮")
                    
                    # Player wins
                    elif (choice1 == "Rock" and king_choice == "Scissor") or \
                         (choice1 == "Paper" and king_choice == "Rock") or \
                         (choice1 == "Scissor" and king_choice == "Paper"):
                        print("\nYou win!💪")
                        player_score_1 += 1
                    
                    # King wins
                    else:
                        print("\nKing of Hearts wins!😕")
                        king_hearts_score += 1
                    
                    # Displaying current score every loop
                    print(f"Score - You: {player_score_1}, King of Hearts: {king_hearts_score}\n")
                
                # Displaying a message for invalid input
                else:
                    print("\nInvalid input! Please choose only one option from Rock, Paper, or Scissor.❎\n")
            
            # Player wins at the end of the round and proceeds to the next stage 
            if player_score_1 > king_hearts_score:
                print("Bravo, adventurer! You're advancing to the next room. Good luck.👀🚪")
                stage_2() # Moving to the next stage of the game
            
            # Player fails and proceeds to the fail function
            else:
                print("♥️ King of Hearts win! ♥️")
                fail()
            
            # Exit the loop if a valid choice is made
            break  

        # Meeting the King of Diamonds if chosen door is middle
        elif "middle" in action1 and "left" not in action1 and "right" not in action1:
            
            # Displaying a message for meeting the King of Diamonds
            print(f"\n{' ♦️' * 60}\n""")
            print("Welcome to the royal adventure! The King of Diamonds awaits you in a high-stake Bluff Card Game.🦹🃏\n")
            input(prompt = "Press <<Enter>> to continue.\n")
            print("""King of Diamonds: Greetings, adventurer! To conquer my realm, you must outwit me in the daring Bluff Card Game.🤡❓ 
Shall you accept the challenge?🫱🏻‍🫲🏾🆙
""")
            input(prompt = "Press <<Enter>> to continue.\n")
            
            # Initializing variables needed for Bluff Card Game
            player_score_2 = 0
            king_diamonds_score = 0

            # Displaying a message for instructions
            print("Welcome to the Bluff Card Game!🤡\n")
            print("""The King of Diamonds will draw a card (1️⃣ to 🔟), and you have to guess if he is bluffing. If the card is 8️⃣ or 
higher, the King is bluffing; otherwise, he's not.❓🦹🤥 Reach a score of two victories before he outsmarts you.
Good luck!
""")
            # Using a loop to play until one player wins 2 rounds
            while player_score_2 < 2 and king_diamonds_score < 2:
                
                # Using a loop to handle player's choice if bluff or not
                while True:
                    
                    # Getting user input for the player's guess
                    player_guess = input(prompt = "\nDo you think the King is bluffing? (yes/no):🤔 ").lower()

                    # Initializing variable for player's guess
                    option2 = ["yes", "no"]
                    
                    # Checking if the player's guess is valid
                    if any(choice.lower() in player_guess.lower() for choice in option2) and \
                        sum(choice.lower() in player_guess.lower() for choice in option2) == 1:
                        player_guess = next(choice for choice in option2 if choice.lower() in player_guess.lower())
                        break
                    
                    # Displaying a message for invalid input
                    else:
                        print("\nInvalid input! Please enter 'yes' or 'no'.🚫\n")

                # King randomly draws a card from 1 to 10 and displaying the drawn card
                king_card = random.randint(1, 10)
                print(f"\nKing's card: {king_card}")

                # Player's guess is correct and the King bluffed
                if (player_guess == "yes" and king_card >= 8) or \
                   (player_guess == "no" and king_card < 8):
                    print(f"You said {player_guess} and that's correct!🤩")
                    player_score_2 += 1
                
                # Player's guess is incorrect and the King bluffed
                elif (player_guess == "no" and king_card >= 8):
                    print("Incorrect! The King was bluffing.😑🤷‍♀️")
                    king_diamonds_score += 1
                
                # Player's guess is incorrect and the King did not bluffed
                else:
                    print("Incorrect! The King wasn't bluffing.😑🤷‍♀️")
                    king_diamonds_score += 1

                # Displaying current score every loop
                print(f"Your score: {player_score_2}, King's score: {king_diamonds_score}\n")

            # Player wins at the end of the round and proceeds to the next stage
            if player_score_2 > king_diamonds_score:
                print("Well done! You've triumphed over the King of Diamonds. Prepare for the next stage of your adventure.🎊👌")
                stage_2() # Moving to the next stage of the game
            
            # Player fails and proceeds to the fail function
            else:
                print("♦️ King of Diamonds win! ♦️")
                fail()
            
            # Exit the loop if a valid choice is made
            break  
            
        # Meeting the King of Spades if chosen door is right
        elif "right" in action1 and "middle" not in action1 and "left" not in action1:
            
            # Displaying a message for meeting the King of Spades
            print(f"\n{' ♠️' * 60}\n""")
            print("""You bravely chose the right door.
Challenge the King of Spades in the thrilling "King Says" game to prove your wit and rule his kingdom.🤫♛\n
""")
            input(prompt = "Press <<Enter>> to continue.\n")
            print("""King of Spades: Step into my kingdom! Prove your worth and conquer me in the "King Says" game.👀👂
You need to at least win two rounds. Are you up for the challenge?🤝⚔️🛡️
""")
            input(prompt = "Press <<Enter>> to continue.\n")
            
            # Defining a list of commands for the King Says Game
            commands = ["King says hi", "teal", "SMILE", "King says JUMP", "purple"]
    
            print("Welcome to King Says!👋🫅🏻")
            print("King will sometimes trick you, so be careful!🤹😜\n")
            print("Get ready...🧏🧐")

            # Adding a delay in display to build suspense
            time.sleep(2)

            # Initializing the player's score and attempt counters
            player_score_3 = 0
            attempt = 0

            # Using a loop to play the game until the player wins or reaches the maximum attempts
            while player_score_3 < 2 and attempt < 2:
                
                king_command = random.choice(commands)
                print(f"\nKing says: {king_command}")

                player_input = input(prompt="Your turn: ").strip().casefold()

                # Validate non-empty input
                while not player_input:
                    print("\nInvalid input. Please enter a valid response.")
                    player_input = input(prompt="Your turn: ").strip().casefold()
                
                # Player's input matches the King's command
                if king_command.casefold() == player_input:
                    print("\nGood job! The King is impressed.👏\n")
                    
                    # Increasing the player's score and resetting the attempt counter
                    player_score_3 += 1
                    attempt = 0
                
                # Player's input did not match the King's command
                elif ("king says" in player_input) and (king_command.lower() not in player_input):
                    print("\nOops! The King tricked you.😆✌️\n")
                    attempt += 1
                
                # Player's input is incorrect
                else:
                    print("\nWrong move!😜\n")
                    attempt += 1
            
            # The King gives a new command for the next round to avoid repetition
            king_command = random.choice(commands)

            # Player wins at the end of the round and proceeds to the next stage
            if player_score_3 == 2:
                print("Hail, victorious adventurer! You've conquered me in the Game of Kings. May luck be your loyal companion!🎉😊")
                stage_2() # Moving to the next stage of the game
                break
            
            # Player fails and proceeds to the fail function
            else:
                print("You lose to the ♠️ King of Spades ♠️")
                fail()
            
            # Exit the loop if a valid choice is made
            break  
        
        # Displaying a message for invalid input
        else:
            print("Invalid input. Please select one (left, middle, right).🙅☝️\n")

#####################################################################################
# Defining a stage_2 function for the second stage of the game
#####################################################################################

def stage_2():
    print(f""" 
{'*' * 120}    

Way to go to the next game!➡️✅
""")
    input(prompt = "Press <<Enter>> to start.\n")
    print("""
Stage 2: Switch Right or Die💡🖲️

    Details: In the room there is one light-bulb and a door to an adjacent room where there are three switches.
    There is one switch that connects to the light-bulb. With the door closed, players may flip any switch. 
    While the door is open, players may only flip the switch once. The door will not close if there are people 
    in both rooms or a switch is flipped.
    
    Rule: It is game clear if players can unanimously answer which switch turns on the light. If the water level
    rises and the surface of the water touches the high current lines, it is game over. 
    """)
    
    input(prompt = "Press <<Enter>> to continue.\n")
    
    # Initializing a variable play_stage2 to True for Stage 2 game
    play_stage2 = True
    
    # Use while loop for Stage 2 question
    while play_stage2:
        
        # Used when user enters a non-numeric value when a numeric value is required
        try:
            result = input(prompt="""Which scenario has the higher probability?. 
        1️⃣ Open the door. Flip any switch with the door open and should it remain off, then guess between the 
            remaining 2 switches.
        2️⃣ Close the door, and flip one switch on until the water rises. Turn off the switch, unclose the door,
            and flip another switch.
        3️⃣ I need a hint.💡
        \n""").lower()

            # Defining lists for different input choices by the player for stage_2
            choice1 = ["open", "open the door", "flip any", "door open", "remain off", "guess", "switches", "1"]
            choice2 = ["close", "close the door", "water", "flip one", "turn", "unclose", "flip another", "2"]
            choice3 = ["I", "need", "hint", "3"]
            result1 = result # To store the chosen choice
            
            # Player choose option 1 or a or any word in choice1 list leading to failure
            if result1 == '1' or result1 == 'a' or any(word in result1.lower() for word in choice1):
                fail()
                play_stage2 = False
            
            # Player choose option 2 or b or any word in choice2 list leading to the next stage  
            elif result1 == '2' or result1 == 'b' or any(word in result1.lower() for word in choice2):
                print("Excellent! The final room is open.") 
                stage_3() # Moving to the next stage of the game
                play_stage2 = False
            
            # Player choose option 3 or c or any word in choice3 list which asks for a hint
            elif result1 == '3' or result1 == 'c' or any(word in result1.lower() for word in choice3):
                print("Hint: Consider the following Python code that might be helpful.🧐")
                print("""
            while water_level < high_current_lines:
                flip_switch_on()
                wait_until_water_rises()
                flip_another_switch()
                print("Is the light bulb hot?")
            """)
                # Calling stage_2 again function when the user wants to have a hint
                stage_2_again()
                play_stage2 = False
            
            # Displaying a message for invalid input
            else:
                print("Invalid entry. Please try again.🚫")
               
        # Handling error when user enters a non-numeric value when a numeric value is required
        except ValueError:
            print("Invalid input. Please enter a valid number.🚫\n")

#####################################################################################
# Defining a stage_2_again function in stage_2 when the user asks for a hint
#####################################################################################

def stage_2_again():
    result = input(prompt = """Which scenario has the higher probability?🚪💡🖲️
        1️⃣ Open the door. Flip any switch with the door open and should it remain off, then guess between the 
            remaining 2 switches. 
        2️⃣ Close the door, and flip one switch on until the water rises. Turn off the switch, open that door, 
            and flip another switch.
        \n""").lower()

    # Defining lists for different input choices by the player for stage_2_again
    choice1 = ["open", "open the door", "flip any", "door open", "remain off", "guess", "switches", "1"]
    choice2 = ["close", "close the door", "water", "flip one", "turn", "unclose","flip another", "2"]
    result1 = result # To store the chosen choice

    # Player choose option 1 or a or any word in choice1 list leading to failure
    if result1 == '1' or result1 == 'a' or any(word in result1.lower() for word in choice1):
        fail()
        play_stage2 = False
            
    # Player choose option 2 or b or any word in choice2 list leading to the next stage
    elif result1 == '2' or result1 == 'b' or any(word in result1.lower() for word in choice2):
        print("Excellent! The final room is open.👏")
        stage_3() # Moving to the next stage of the game
        play_stage2 = False
    
    # Displaying a message for invalid input   
    else:
        print("Invalid entry. Please try again.🚫\n")
        stage_2_again() # Calling the function stage_2_again if the input is invalid

#####################################################################################
# Defining a stage_3 function for the third stage of the game
#####################################################################################

def stage_3():
    random.seed(None) # Setting a random seed for consistent results
    print(f""" 
{'*' * 120}    

Can you make it to the final round?👹🤓
""")

    input(prompt = "Press <<Enter>> to continue.\n")
    print("""
Stage 3: Beauty Contest💄🧖🏻‍♀️🤖

    Details: Player must choose a number between 1️⃣ and 🔟. The computer will also make a guess.🤖🧠🤔\n
    Rule: The winner is determined based on the absolute difference between the player's guess and a randomly 
    generated number, with the goal of having a smaller difference than the computer.🧮🤯🔢
    """)
    
    # Asking the player for input until a valid number between 1 and 10 is entered
    while True:
        
        # Handling potential errors during user input in the stage_3 function
        try: 
            guess = int(input(prompt = "Pick a number between 1️⃣ and 🔟 (whole number only): "))

            # Checking if the input is within the specified range
            if 1 <= guess <= 10:
                break # If valid, break out of the loop
            
            # Displaying a message for invalid input
            else:
                print("Number must be between 1 and 10. Please try again.🚫\n")

        # Handling the ValueError exception if the input is not an integer
        except ValueError:
            print("Invalid input. Please enter a valid number.🚫\n")

    print(f"You entered: {guess}")

    # Generating a random number for the computer's guess and calculate the differences
    number = random.randint(a=1, b=10)
    
    # Calculating the absolute difference between the player's guess and the computer's random number
    difference = abs(guess - number)

    # Calculating the absolute difference between the player's guess and the previously calculated difference
    player_difference = abs(guess - difference)

    # Adjusting the computer's guess based on the difference between the player's guess and a random number
    if difference > 2:
        computer_guess = random.randint(1, 10) # The computer randomly guesses a number between 1 and 10
    
    # Conditional statement that adds a random number between -1 and 1 to the player's guess
    else:
        computer_guess = guess + random.randint(-1, 1) 

    # Calculating the absolute difference between the computer's guess and the original difference 
    computer_difference = abs(computer_guess - difference)

    print(f"""
Your number is {guess} and the computer guessed {computer_guess}.
The difference is {difference}.
    """)
    
    # Player wins
    if player_difference <= computer_difference:
        win() # Calling the win function if player wins
    
    # Computer wins
    else:
        print("Sorry, you've failed. Game over.💀⚰️🩸🧛\n")
        fail() # Calling the fail function when the player loses to computer

# Defining a fail function when the player fails
def fail():
    print(f""" 
{'*' * 120}  

Tough luck, my friend. The room swallowed you whole.😵🚨
Better luck next time... maybe in another life!🤷‍♂️💀

          ==============================
          ||     ||<(.)>||<(.)>||     ||
          ||    _||     ||     ||_    ||
          ||   (__D     ||     C__)   ||
          ||   (__D     ||     C__)   ||
          ||   (__D     ||     C__)   ||
          ||   (__D     ||     C__)   ||
          ||     ||     ||     ||     ||
          ==============================
          
""") 
    
    # Calling the play_again function to confirm if the player wants to play again
    play_again() 

# Defining a win function when the player wins
def win():
    print("""
    
██    ██  ██████  ██    ██     ███████ ██    ██ ██████  ██    ██ ██ ██    ██ ███████ ██████  ██ 
 ██  ██  ██    ██ ██    ██     ██      ██    ██ ██   ██ ██    ██ ██ ██    ██ ██      ██   ██ ██ 
  ████   ██    ██ ██    ██     ███████ ██    ██ ██████  ██    ██ ██ ██    ██ █████   ██   ██ ██ 
   ██    ██    ██ ██    ██          ██ ██    ██ ██   ██  ██  ██  ██  ██  ██  ██      ██   ██    
   ██     ██████   ██████      ███████  ██████  ██   ██   ████   ██   ████   ███████ ██████  ██ 
                                                                                                


You just bagged ¥20,000,000, roughly $132,000!🏆🎖️🤑🤑🤑
You made it out alive!🗝️👏😊
""")
    
    # Calling the play_again function to confirm if the player wants to play again
    play_again() 

#####################################################################################
# Defining a play_again function to ask if the player wants to play again
#####################################################################################

def play_again():
    input(prompt = "Press <<Enter>> to exit.\n")
    
    # Using a loop for the play again input prompt
    while True:
        
        # Initializing variables if the player wants to play again
        option3 = ["yes", "no"]
        new_game = input(prompt = "Do you want to play again? (yes/no): ").lower()
        
        # Checking if the player's choice is valid
        if any(play_ask.lower() in new_game.lower() for play_ask in option3) and \
            sum(play_ask.lower() in new_game.lower() for play_ask in option3) == 1:
            new_game = next(play_ask for play_ask in option3 if play_ask.lower() in new_game.lower())            

            # Player choose yes to play again
            if new_game == 'yes':
                game_intro() # Calling the game_intro function if the player wants to play again
                break # To exit the loop after calling game_intro function

            # Player choose not to play again
            elif new_game == 'no':
                print("""\nThank you for taking on the challenge in the Python Escape Room Game.🙂👑💂‍♂️

Copyright © 2023 Python – All Rights Reserved.
""")
                # To exit the loop
                break 

        # Displaying a message for invalid input
        else:
            print("Invalid input. Please enter 'yes' or 'no'.\n")
            
# Calling the game_intro function to start the game
game_intro()

```

References <br><br>
Ayberk, Sony (2023, November). List Comprehensions in Python and Generator Expressions.<br>&nbsp;&nbsp;&nbsp;&nbsp;https://djangostars.com/blog/list-comprehensions-and-generator-expressions/<br>
Beauty Contest (Season 2, Episode 6)  [TV series episode]. (2022). In Alice in Borderland. Netflix <br>
Dead or Alive (Season 1, Episode 1)  [TV series episode]. (2020). In Alice in Borderland. Netflix <br>
Fandom. (n.d.). Four of Diamonds (Netflix). Alice in Borderland Fandom. https://aliceinborderland.fandom.com/wiki/Four_of_Diamonds_(Netflix)<br>
Fandom. (n.d.). King of Diamonds (Netflix). Alice in Borderland Fandom. https://aliceinborderland.fandom.com/wiki/King_of_Diamonds <br>
Fandom. (n.d.). Three of Clubs (Netflix). Alice in Borderland Fandom. https://aliceinborderland.fandom.com/wiki/Three_of_Clubs_(Netflix)#Participants <br>
Federico Furzan. (2023, August). Escape Room Ending, Explained. https://movieweb.com/escape-room-ending-explained/ <br>
Light Bulb (Season 1, Episode 5) [TV series episode]. (2020). In Alice in Borderland. Netflix <br>
OpenAI. (2023). ChatGPT (Mar 14 version) [Large language model]. https://chat.openai.com/chat<br>
SK Desk. (2023, January). What is Alice in Borderland about?. https://www.sportskeeda.com/pop-culture/what-is-alice-in-borderland-about<br>
Stackoverflow. (2011, August). https://stackoverflow.com/questions/7006839/alternative-to-list-comprehension-if-there-will-be-only-one-result<br>
Stackoverflow. (2012, March). Find a a value in a list. https://stackoverflow.com/questions/9542738/find-a-value-in-a-list/9542768#9542768<br>
Stackoverflow. (2022, February). List Comprehension Method. 
<br>&nbsp;&nbsp;&nbsp;&nbsp;https://stackoverflow.com/questions/71185614/take-exact-number-of-inputs-using-list-comprehension-method-on-python

Copyright © 2023 Python – All Rights Reserved.
