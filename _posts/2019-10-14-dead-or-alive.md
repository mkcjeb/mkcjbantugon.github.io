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

This game  is based on the first episode of the tv series The Walking Dead (2010) and has some references from the movie Zombieland (2009). More specifically, line 805 is a direct quote from The Walking Dead, episode 01 - season 01 min 1:05:00. The explanation from how the zombie virus was created is a reference to Zombieland.
    
The game consists of four stages, and also has defined functions for set ups in each stage, user inventory, alive and dead. Certain codes were inspired by Professor Chase Kusterer from Hult International Business School (example games - line 813).
    
This game requires courage, and having a small amount of survival knowledge in case of a zombie apocalypse will be helpful.
    
1. Round 1: Hospital;
2. Round 2: Home;
3. Round 3: On the road to Atlanta;
4. Round 4: Atlanta.

## Instructions

To run the game, I suggest copy-pasting the following code block into a Jupyter Notebook code cell (I know the code is long, but don't worry about it, we just need all of these things to make it awesome). Then, you only need to run the cell to start the game and follow the printed commands!

Have fun with it, and bonus points for those who can find the easter eggs!


*Note: If you're not familiar with Jupyter Notebooks, here's a [great tutorial](https://www.youtube.com/watch?v=HW29067qVWk) on how they work and [how to set up](https://www.anaconda.com/products/individual) Anaconda Navigator.*


```python
"""
Created on Mon Oct 14, 2019

@author: sophie.briques
"""

"""
    A) Introduction:
    This game is based on the first episode of the tv series The Walking Dead
    (2010) and has some references from the movie Zombieland (2009).
    More specifically, line 805 is a direct quote from The Walking Dead,
    episode 01 - season 01 min 1:05:00. The explanation from how the zombie virus 
    was created is a reference to Zombieland.
    
    The game consists of four stages, and also has defined functions
    for set ups in each stage, user inventory, alive and dead.
    
    Certain codes were inspired by Professor Chase's example games (line 813).
    
    This game requires courage, and having a small amount of survival
    knowledge in case of a zombie apocalypse will be helpful.
    
    Round 1: Hospital;
    Round 2: Home;
    Round 3: On the road to Atlanta;
    Round 4: Atlanta.
        
    B) Bugs and Errors:
   -  User name input: if statement does not run 


"""

##############################################################################

##Importing necessary packages
from sys import exit
from random import randint
import time

##Waking up the user
def setup():
    print("""
Hey... hellooo? can you hear me? .... is there anyone there?
""")
    time.sleep(2)
    
    print("""
Hello? Come on... you need to wake up, like, *RIGHT NOW*!
""")
    
    input("""<Press enter to show you're you awake>\n""")
    print("""
Conscience: Oh! Good. Hi there. I'm your conscience. How do you feel?""")
    start_game()

## First user interaction, introducing user's conscience
def start_game():
    feeling = input("""
    
a) Confused... what is going on?
b) Am I still sleeping?
c) Perfectly fine, thank you for asking kind voice in my head.

""").lower()
    
    ## User selects a
    if feeling == 'a' or feeling == '1' or feeling == 'Confused':
        time.sleep(1)
        print("""
Conscience: Don't worry, everything is ok, we'll get through this together!
            You just woke up from a coma.
            It is normal to feel this way.
            
            Do you remember anything?
""")
        time.sleep(2)
        print("""
Conscience: Oh yeah true... I guess not. Stupid me!
            That blow to your head must have been really strong.
""")
        time.sleep(2)
        print("""
Conscience: Let's start with the basics then, shall we?\n
""")
        ## Bringing user to first scene
        waking_up()
    
    ### User selects b
    elif feeling == 'b'or feeling == '2' or 'am i still sleeping?' in feeling:
        time.sleep(1)
        print("""
Conscience: NO! THIS IS REAL LIFE! Snap out of it! You just woke up from a coma.
            ugh.. you don't remember anything do you? This will be fun... \n
""")
        time.sleep(2)
        print("""
Conscience: That blow to your head must have wiped out most of your memory.
            Let's just hope you still remember how to RUN - you'll need it. \n
""")
        time.sleep(2)
        print("""
Conscience: Alright, let's start with the basics. \n
""")
        
        ## Bringing user to first scene
        waking_up()
    
    ### User selects c
    elif feeling == 'c' or feeling == '3' or feeling == 'fine'or feeling == 'perfectly fine':
        time.sleep(1)
        print("""
Conscience: No problem buddy! Do you remember anything at all? 
            Of what happened to you? No? It's alright.
            You suffered from a pretty big blow to your head.
""")
        time.sleep(4)
        print("""
Conscience: Let's see if you can at least recall some of the basics shall we? \n
""")
        ## Bringing user to first scene
        waking_up()
    
    ### Buffering any mistakes in user input
    else:
        time.sleep(1)
        print("Invalid entry. Please try again.\n")
        start_game()


## First scene - Defining essential variables and story line
def waking_up():

    ## Defining global variables (the basics)
    global your_name
    global your_SO

    ## Defining user's name
    time.sleep(2)
    your_name = input("""
Conscience: Now, do you remember your name?\n
    > Input what you think your name is:
""").capitalize()
    if your_name == "no":
        input("""
Conscience: Nope, thata's not right... try again!
""")
        waking_up()
    elif your_name == " ":
        input("""
Conscience: Nope, thata's not right... try again!
""")
        waking_up()
    elif your_name == "yes":
        input("""
Conscience: Nope, thata's not right... try again!
""")
        waking_up()

    ##Defining significant other's name
    time.sleep(1)
    print(f"""
Conscience: Yes, uff!, you got that right, it's not as bad as I thought.
            You're definetly {your_name}!""")
    
    time.sleep(2)
    your_SO = input(f"""\n
Conscience: Now, let's see now if you remember your family's name.
            What's the name of your significant other? \n
> Input what you think the name of your significant other is:
""").capitalize()
    if your_SO == "no":
        input("""
Conscience: Nope, thata's not right... try again (it's ok to be creative)!
""")
    elif your_SO == " ":
        input("""
Conscience: Nope, thata's not right... try again (it's ok to be creative)!
""")
    elif your_SO == "yes":
        input("""
Conscience: Nope, thata's not right... try again (it's ok to be creative)!
""")
    elif your_SO == "single":
        input("""
Conscience: Nope, thata's not right... try again (it's ok to be creative)!
""")
        
        waking_up()
    
    time.sleep(2)
    input(f"""\n
Conscience: Yes! {your_SO}!!! Oh my, I do hope {your_SO} is safe... 
            Hm.. What?... Oh, nevermind about that right now.
            We need to get you up and running. \n
<Press enter to continue>\n
""")
    time.sleep(1)
    hospital_setup()

## end of game set up ##

############## ROUND 1 - Hospital ##############
## In this round, the user will atempt to escape the abandoned hospital without
## being eaten by zombies that have been kept in a section of the hospital.
## The options in this round are randomized as our user does not know yet
## where they are or what is going on.

#### Setting the scene of the hospital - Something is going on ####
def hospital_setup():
    print(f"""\n
Conscience: Ok {your_name}, now listen carefully.
            
            You are in the hospital because of a head injury that left you in a coma.
            I'm glad you remember everything.
            However, I think somtehing is wrong.
            It's been very quiet around here.
            The flowers near your bed are dead, and the clock is stopped.
""")
    
   
    time.sleep(1)
    print('-' * 50)
    input("""
  < Press enter to call the nurse > \n
""")
    time.sleep(1)
    print('no one answers...\n')
    print('-' * 50)

    time.sleep(2)
    print("""... ok, this is sketchy, we need to get out of here... and FAST.""")
    
    ## Taking user to hospital challenge
    hospital_challenge()
    
#### Abandoned hospital Challenge - GOAL : Get out alive ####
def hospital_challenge():
    input('< Press enter to leave your room and walk down the hallway >')
    time.sleep(1)
    print(f"""
Conscience: {your_name}! I see the exit. It's over there at the end of the hallway. 
""")
    time.sleep(2)
    print("""
Conscience: Oh no wait, it might be this one on the right.
""")
    time.sleep(2)
    print("""
Conscience: Uh... what about the one in the left?
""")
    time.sleep(1)
    ## Defining random doors with zombies
    door = input("""
- Which door do you choose to exit the hospital?
         a) End of hallway
         b) Right
         c) Left
""").lower()
    zombies = randint(0,2)
    
    ## Start of conditional statement for round 1 - if user selected a
    if "a" in door or "end of hallway" in door or "end" in door or "1" in door or "hallway" in door:
        door_int = 0
        
        ### Start of nested conditional to include randomization of zombies in doors
        if zombies != door_int:
            time.sleep(1)
            print(f"""
Conscience: Uff! Escaped this freakish place. What is going on?? Let's try to find out {your_name}.
""")
            time.sleep(2)
            home_setup()
        elif zombies == door_int:
            print(f"""
Conscience: Are you hearing this? Are these...
""")
            time.sleep(1)
            print(f"""
Conscience: Growling noises???
""")
            time.sleep(2)
            dead()
        else:
            time.sleep(1)
            print("""
Something went wrong. Please try again
""")
            time.sleep(1)
            hospital_challenge() 
        ### end of nested conditional
    
    ## Continuing of conditional statement for round 1 - if user selected b
    elif "b" in door or "right" in door or "r" in door or "2" in door:
        door_int = 1
        
        ### Start of nested conditional to include randomization of zombies in doors
        if zombies != door_int:
            time.sleep(1)
            print(f"""
Conscience: Uff! Escaped this freakish place. What is going on?? Let's try to find out {your_name}.
""")
            time.sleep(1)
            home_setup()
        elif zombies == door_int:
            time.sleep(1)
            print(f"""
Conscience: Are you hearing this? Are these...
""")
            time.sleep(1)
            print(f"""
Conscience: Growling noises???
""")
            time.sleep(1)
            dead()
        else:
            time.sleep(1)
            print("""
Something went wrong. Please try again
""")
            hospital_challenge() 
        ### end of nested conditional
    
    ## Continuing of conditional statement for round 1 - if user selected c
    elif "c" in door or "left" in door or "l" in door or "3" in door:
        door_int = 2
        
        ### Start of nested conditional to include randomization of zombies in doors
        if zombies != door_int:
            time.sleep(1)
            print(f"""
Conscience: Uff! Escaped this freakish place. What is going on?? Let's try to find out {your_name}.
""")
            time.sleep(1)
            home_setup()
        elif zombies == door_int:
            time.sleep(1)
            print(f"""
Conscience: Are you hearing this? Are these...
""")
            time.sleep(1)
            print(f"""
Conscience: Growling noises???
""")
            time.sleep(1)
            dead()
            
        else:
            time.sleep(1)
            print("""
Something went wrong. Please try again
""")
            time.sleep(1)
            hospital_challenge() 
        ### end of nested conditional
    
    else:
        time.sleep(1)
        print("""
Conscience: What sorry? Did you forget which way was right or left?
            Really? Come on you can do this. Try again.""")
        time.sleep(1)
        hospital_challenge() 
            

            
############## ROUND 2 - Home ##############
## In this round, the user just made it home from the hospital on a bike.
## The user will not find his/her family at home. He(r) will however meet the
## neighbour, who will explain what is going on.
## Then, the user, after deciding to go to Atlanta to find his/her family, 
## will have to collect into the inventory, items that will help achieve the goal.
## There are no active zombies in this cage that can kill the user, so everyone 
## should pass. There is a. mini easter egg in it! Have fun!

#### Setting the scene of home - nobody is there ####
def home_setup():
    time.sleep(1)
    print("""
- You just walked outside the hospital into the parking lot. - 
""")
    time.sleep(1)
    input(f"""
Conscience: {your_name} is that smell coming from you?
            Gee, you should think about taking a shower from time to time...
            I mean, it's really hard to think in these conditions...\n
< Press enter to make your conscience shut up > \n
""")
    time.sleep(1)
    print(f"""Sorry {your_name}.""")
    time.sleep(2)
    input("""
Conscience: Wait...
            What are those white bags??

< Press enter to get closer and inspect >""")

    input("""
- You are looking at body bags spread all over the hospital's parking lot.
  The bodies are deformed, some with big pieces of flesh missing,
  all with drooping skin. - \n 
< Press enter to continue >
""")
    time.sleep(2)
    input(f"""
Conscinence: {your_name} I think we need to get home as soon as possible.
             Something is not right. It looks like a ghost town around here.
             Let's grab that bike! \n
< Press enter to hop on the bike and go home > \n
""")
    
    time.sleep(3)
    
    ## taking user home for neighbour's explanation
    home_neighbour()

### Conversation with neighbour ###
def home_neighbour():
    time.sleep(1)
    print(f"""
- You arrive home, the door is open wide. You enter the house. -
""")
    time.sleep(3)
    in_house = input("""
- You are trying to find your family. Do you wish to:
        a) call out for their names
        b) go into every room, you don't know if there is someone else in the house.
""").lower()
    
    ##Start of conditional statement
    if in_house == "a" or in_house == "call out" or in_house == "1" or in_house == "call out for their names":
        input("""
Conscience: SHHHHHHHH! What if... what if.... there is one of those here?
            Maybe try going into every room carefully...

< Press enter to continue >
""")
    elif in_house == "b" or in_house == "go into room" or in_house == "2" or in_house == "go into every room":
        time.sleep(1)
        print(f"""
Conscience: You know what {your_name}, that's a good idea.
            I see your judgement has not been affected by the head blow.""")
   
    ## Buffer in case user gets a wrong answer --> a mini easter egg ;) 
    else:
        cry = input(f"""
Something went wrong. Please try again by entering one of the two options. 
Unless you want to sit and cry for 5 seconds. That's ok too.\n
 > Just enter "cry" if that's the case, if not, just enter "try again" 
""")
        if cry == "cry":
            time.sleep(5)
            home_neighbout()
        elif cry == "try again":
            time.sleep(1)
            home_neighbour()
        else:
            time.sleep(1)
            print(f"""
Conscience: Come on {your_name}, I know you can do this. Your coma was not THAT long.
            Here's a little help:
            Enter the letter "a" or "b" when asked what to do to find your family.""")
            time.sleep(1)
            home_neighbour()
    
    ## End of conditional statement
    time.sleep(2)
    print(f"""
- It seems like {your_SO} is not here. -
""")
    time.sleep(1)
    print("""
- You turn around and there is a man behind you, alive and well...
  pointing a gun at you... - """)
    time.sleep(1)
    print("""
- You suddenly recognize its your neighbour, Kyle, and you both fall into an embrace. - 
""")
    input("""
- Press enter to ask Kyle what is going on - 
""")
    time.sleep(1)
    print(f"""
Kyle: Oh {your_name}! We all thought you had died after the hospital was taken over.
      The world as we know it is over.
      
      A few months after you entered your coma, a certain strain of the mad cow disease
      mutated into a human virus, which, when contaminated, killed people with a
      terrible fever, only to make them come back as zombies.

""")
    time.sleep(5)
    print(f"""
{your_name}: What about {your_SO}?
""")
    time.sleep(1)
    print(f"""
Kyle: I don't know {your_name}. I haven't seen {your_SO} since it all started.
      A lot of people left to Atlanta.
      Apparently there is a center with no dead there.
      That's the only place I can think of.
""")
    time.sleep(2)
    input(f"""
Conscience: We should go to Atlanta then, we need to find {your_SO}.
            Let's gather some supplies. \n
< Press enter to continue >
""")
    time.sleep(1)
    input(f"""
Kyle: Oh and {your_name}, be careful with noise and light.
      
      These things will not stop until they find where it comes from... 
      Especially when they are hungry... \n
< Press enter to continue >
""")
    time.sleep(1)
    inventory()

### Gathering supplies for the trip ###
def inventory():
    
    # Defining the inventory empty list
    inventory_lst = []
    choices_lst = ['a) Family Photo Albums',
                   'b) Clothes',
                   'c) 5 Water bottles and 5 Canned foods',
                   'd) Gun and bullets',
                   'e) Match Box',
                   'f) Big Kitchen knife']

    ## start of while loop for 3 item choices to put in the bag
    print("""
Conscience: Alright, here are a few items you can choose from.
            
            BE CAREFUL - You can choose 3 items to put in your duffle bag.
            We don't want anything slowing us down:""")
    items = 3       
    while items > 0:
        print(*choices_lst, sep='\n')

        choices = input('> \n').lower()

        if choices == "a" or choices == "1" or choices == "family photo albums" or choices == "photos":
            inventory_lst.append("Family Photo Albums")
            choices_lst.remove("a) Family Photo Albums")
        elif choices == "b" or choices == "2" or choices == "clothes":
            inventory_lst.append("Clothes")
            choices_lst.remove("b) Clothes")
        elif choices == "c" or choices == "3" or choices == "water and food" or choices == "food" or choices == "5 water bottles and 5 canned foods":
            inventory_lst.append("5 Water bottles and 5 Canned foods")
            choices_lst.remove("c) 5 Water bottles and 5 Canned foods")
        elif choices == "d" or choices == "4" or choices == "gun" or choices == "gun and bullets":
            inventory_lst.append("Gun and bullets")
            choices_lst.remove("d) Gun and bullets")
        elif choices == "e" or choices == "5" or choices == "matches" or choices == "match box" or choices == "box":
            inventory_lst.append("Match Box")
            choices_lst.remove("e) Match Box")
        elif choices == "f" or choices == "6" or choices == "knife" or choices == "big knife" or choices == "kitchen knife":
            inventory_lst.append("Big Kitchen knife")
            choices_lst.remove("f) Big Kitchen knife")
        else:
            print("""
        Conscience: Oh it appears the item you are trying to collect is no longer in your home...
                    Someone must have stolen it! Sorry, try again. """)
            input('<Press enter to try again>')
            inventory()
        
        
        items -= 1
    
    time.sleep(1)    
    print(f"""\n
    Great! {your_name} you now have {inventory_lst[0]}, {inventory_lst[1]}
    and {inventory_lst[-1]} in your inventory!""")
    
    # Taking the user to round 3: on the road to Atlanta
    time.sleep(2)
    on_the_road()            
            

############## ROUND 3 - On the Road to Atlanta ##############
## In this round, the user took his/her car to go to Atlanta and try to find
## his/her family. The user will encounter a home along the way, but the longer
## he/she goes, the gas meter goes down noticeably. The user will have to decide
## whether or not to stop at the house to ask for gas. If him/her chooses
## not to stop, they will be stranded on the road and be killed by zombies.
## If they stop, they will be faced with the option to go inside and ask for gas
## or take the horse and make the rest of the way by horse. The house is full
## of zombies that will kill the user, and the horse will lead the user to the
## final round.

#### Setting the scene of getting on the road - where are you going to get gas? ####

## try to add the inventory inside this function
def on_the_road():
    input(f"""
Conscience: Alright {your_name}, now that we have all we need, let's hit the
            road to find {your_SO}.
            Good thing your car is still here. \n
< Press enter to get in the car and start on the road to Atlanta >
""")
    time.sleep(2)
    gallons = 4
    print("""
- You are on the highway to Atlanta -  
""")
    time.sleep(2)
    print(f"""
Conscience: What? We only have {gallons} gallons of gas? Oh no... this is no good.
            We can't make it to atlanta on that! \n
""")
    
    time.sleep(2)
    exit = input("""
Conscience: I see an exit! Do you want to take the exit or keep going?
                a) Take exit
                b) Keep driving
""").lower()
        
        ## Conditional to take the exit
    if "a" in exit or "exit" in exit or "take the exit" in exit or "1" in exit:
        print("""
- You have taken the exit - 
""")
        gallons -= 1
        time.sleep(2)
        stop_house = input("""
Conscience: I see a house! This is a nice little farm... Let's stop here.
            Look! They even have a horse!
            What do you want to do?
                a) Knock on door and ask for gas
                b) Say hi to the horse
""").lower()
            
            ## Nested conditional to go into the house or steal the horse
        if "a" in stop_house or "knock" in stop_house or "1" in stop_house or "knock on door and ask for gas" in stop_house or "ask for gas" in stop_house:
            time.sleep(2)
            print("""
Conscience: Hm it seems like no one is home.
""")
            time.sleep(2)
            print("""                
Conscience: Oh no, are you hearing this? Are these growling noises?
""")
            dead()
          
        elif "b" in stop_house or "say hi" in stop_house or "2" in stop_house or "horse" in stop_house or "say hi to the horse" in stop_house:
            time.sleep(1)
            print("""
Conscience: It seems like the horsy likes us!
""")
            time.sleep(1)
            input(f"""
Conscience: You know {your_name}, it wouldn't be a bad idea to use him to take us
            to Atlanta.
            
            At least he will never run out of gas!
            And the house looks empty, I'm sure nobody is here...

            
< Press enter to saddle the horse and hop on the road again to head to Atlanta >
""")
            atlanta()
        else:
            input("""
Something went wrong. Please press enter to try again.
Remember! Input the letter of the option you would like to choose""")
            on_the_road()
          
          ## end of nested conditional
    elif "b" in exit or "keep going" in exit or "keep driving" in exit or "keep" in exit:   
        
        ## while loop for gas left until Atlanta
        miles_to_atlanta = 40 * 2
        while gallons > 0 and miles_to_atlanta > 0:
            gallons -= 1
            miles_to_atlanta -= 20
            
            # start of conditional
            if gallons > 0 and miles_to_atlanta >0:
                time.sleep(2)
                print(f"""
- You have {gallons} gallons of gas left -          
""")
            elif gallons > 0 and miles_to_atlanta == 0:
                print(f"""
- Congratulations! You have arrived -          
""")
            else: #gallons == 0 and miles_to_atlanta > 0:
                time.sleep(2)
                print("""
- You are out of gas and in the middle of a highway. - """)
                time.sleep(3)
                print("""
Conscience: You know what, it will be ok. Let's just keep walking. 
""")
                time.sleep(2)
                print("""
Conscience: Oh no, are you hearing this? Are these growling noises?
""")
                time.sleep(1)
                dead()
            
            ## end of while loop
        
    else: 
        input("""
Something went wrong. Please try again.
Remember! Input the letter of the option you would like to choose""")
        on_the_road()

############## ROUND 4 - Atlanta - FINAL ROUND ##############
## In this round, the user is in Atlanta, trying to find the zombieless paradise,
## hoping the family will be there. However, in the city the user will find a 
## hungry mob of zombies (regardless of directions the user takes) instead
##that will attack. With sharp decisions, the user can get out alive, and
## find a radio that will let him or her talk to their relatives.
##### this round has an easter egg - good luck! #####

#### Setting the scene of Atlanta - where is everybody? ####
def atlanta():
    zombie_free = input(f"""
Conscience: {your_name}, we are finally in Atlanta! Oh my I can feel our heart 
            beating fater to think we're so close to finding {your_SO}. 
        
Conscience: Let's try to find the zombie free center. Where do you think that is?
                a) Airport
                b) CDC Headquarters
""").lower()
    
    time.sleep(1)
    
    ## Start of conditional
    if "a" in zombie_free or "1" in zombie_free:
        print("""
Conscience: Ok! I can see the sign for the Airport! It says to take a right here! 
""")
        time.sleep(1)
        input("""
< Press enter to take a right on the next block >
""")
    elif "b" in zombie_free or "2" in zombie_free or "c" in zombie_free or "h" in zombie_free:
        print("""
Conscience: Ok! I remember the CDC is a little further away! Let's take a left here! 
""")
        time.sleep(1)
        input("""
< Press enter to take a left on the next block >
""")
    
    else:
        input("""
Something went wrong. Please press enter to try again.
Remember! Input the letter of the option you would like to choose""")
        atlanta()
    ## End of conditional

## Encountering the mob of zombie
    time.sleep(2)
    print("""\n      
 ▄▄▄       ██▀███    ▄████  ██░ ██  ██░ ██ 
▒████▄    ▓██ ▒ ██▒ ██▒ ▀█▒▓██░ ██▒▓██░ ██▒
▒██  ▀█▄  ▓██ ░▄█ ▒▒██░▄▄▄░▒██▀▀██░▒██▀▀██░
░██▄▄▄▄██ ▒██▀▀█▄  ░▓█  ██▓░▓█ ░██ ░▓█ ░██ 
 ▓█   ▓██▒░██▓ ▒██▒░▒▓███▀▒░▓█▒░██▓░▓█▒░██▓
 ▒▒   ▓▒█░░ ▒▓ ░▒▓░ ░▒   ▒  ▒ ░░▒░▒ ▒ ░░▒░▒
  ▒   ▒▒ ░  ░▒ ░ ▒░  ░   ░  ▒ ░▒░ ░ ▒ ░▒░ ░
  ░   ▒     ░░   ░ ░ ░   ░  ░  ░░ ░ ░  ░░ ░
      ░  ░   ░           ░  ░  ░  ░ ░  ░  ░
                                           
      \n""")
    
    time.sleep(2)
    print("""
- You encounter a mob of zombies - 
""")
    time.sleep(1)
    print("""
Conscience: They look hungry...
""")
    time.sleep(1)
    print("""
- Your horse starts to freak out, you loose your inventory bag in the process -
""")
    time.sleep(2)
    final()

def final():
## Start of final decision to win the game
    final_decision = input("""
Conscience: QUICK!!!! 
            What do you want to do?
                a) Outrun them - they don't look like they can run that fast
                b) There's a building across the street - try to go in
                c) Crawl underneath the abandoned military tank
                
""").lower()
    
    ## Start of conditional
    
    ### User chooses to outrun them - ha ha
    if "a" in final_decision or "1" in final_decision or "run" in final_decision:
        run = 5
        while run > 0:
            input ("""
< Press enter to run >
""")
            run -= 1
        print(f"""
Conscience: {your_name}, I think there are too many of them...        
""")
        time.sleep(2)
        dead()
    
    ### User chooses to go inside the building - ops
    elif "b" in final_decision or "building" in final_decision or "in" in final_decision or "street" in final_decision or "2" in final_decision:
        print(f"""
Conscience: Oh no... the door is locked. Nice work {your_name} ... Genius.
""")
        run = 5
        while run > 0:
            input ("""
< Press enter to run the other way >
""")
            run -= 1
        print(f"""
Conscience: {your_name}, I think there are too many of them...        
""")
        time.sleep(2)
        dead()
    
    ### User chooses to crawl under the tank - YES
    elif "c" in final_decision or "3" in final_decision or "crawl" in final_decision or "under" in final_decision or "tank" in final_decision:
        crawl = input(f""" 
Conscience: Are you really sure this is a good idea? \n
             a) YES! SHUT UP BRAIN!
             b) No... you're right, we can outrun them! We go this!

""").lower()
        ## Start of nested conditional
        if "a" in crawl or "1" in crawl or "yes shut up brain" in crawl or "y" in crawl:
            time.sleep(2)
            print("""
- You crawl under the tank. There are zombies crawling behind you. - 
""")
            time.sleep(1)
            print("""
- You keep going but as you move forward, zombies are crawling under the other side as well. -
""")
            time.sleep(1)
            print("""
- Desperately, you look up to to say your last words, and you see the under door to the tank.
  You pull yourself up and close the door behind you. You are now in the tank. -
""")
            time.sleep(1)
            print("""
- You pull yourself together trying to catch your breath, when the radio starts making a noise: 

Radio: "Hey you? Dumbass. Yeah you in the tank. Cozy in there?" \n
""")
            time.sleep(2)
            alive()
            
        elif "b" in crawl or "2" in crawl or "run" in crawl or "no" in crawl or "right" in crawl:
            run = 5
            while run > 0:
                input ("""
< Press enter to run the other way >
""")
                run -= 1
            print(f"""
Conscience: {your_name}, I think there are too many of them...        
""")
            dead()
        else: ## try to have it back to the middle of questions 
            print(f"""
Conscience: WE DON'T HAVE TIME FOR THINKING {your_name} CLOWN.
            WE NEED TO MAKE A DECISION NOW IF YOU WANT TO SEE {your_SO} AGAIN.
""")
            time.sleep(1)
            input("""
< Press enter to try to make a decision again and not be eaten by zombies > 
""")
            time.sleep(2)
            final()
        
    ## easter egg
    elif final_decision == "jeep":
        print("""
Conscience: Great idea! Use the radio from the military jeep in the corner.
""")
        time.sleep(2)
        print("""
- After entering the jeep to use the radio, you realize the key is in the ignition.
  
  You turn the ignition on and the car starts. SUPRISE: the tank is full.
  
  You use the car to leave the mob behind (not before stopping near your bag of inventory
  and taking it with you.) - 
""")
        time.sleep(2)
        alive()
    
    ## Buffer for mistakes
    else:
        time.sleep(2)
        print(f"""
Conscience: WE DON'T HAVE TIME FOR THINKING {your_name} CLOWN.
            WE NEED TO MAKE A DECISION NOW IF YOU WANT TO SEE {your_SO} AGAIN.
""")
        time.sleep(1)
        input("""
< Press enter to try to make a decision again and not be eaten by zombies > 
""")
        time.sleep(2)
        final()
    ## end of conditional

##############################################################################
############## Dead Function ##############
## This function defines any time where the user is killed by zombies.
## It gives the option to play again, as the game has random aspects to it.

def dead():
    print(40 * "-")
    time.sleep(2)
    print("""
▒███████▒ ▒█████   ███▄ ▄███▓ ▄▄▄▄    ██▓▓█████   ██████ 
▒ ▒ ▒ ▄▀░▒██▒  ██▒▓██▒▀█▀ ██▒▓█████▄ ▓██▒▓█   ▀ ▒██    ▒ 
░ ▒ ▄▀▒░ ▒██░  ██▒▓██    ▓██░▒██▒ ▄██▒██▒▒███   ░ ▓██▄   
  ▄▀▒   ░▒██   ██░▒██    ▒██ ▒██░█▀  ░██░▒▓█  ▄   ▒   ██▒
▒███████▒░ ████▓▒░▒██▒   ░██▒░▓█  ▀█▓░██░░▒████▒▒██████▒▒
░▒▒ ▓░▒░▒░ ▒░▒░▒░ ░ ▒░   ░  ░░▒▓███▀▒░▓  ░░ ▒░ ░▒ ▒▓▒ ▒ ░
░░▒ ▒ ░ ▒  ░ ▒ ▒░ ░  ░      ░▒░▒   ░  ▒ ░ ░ ░  ░░ ░▒  ░ ░
░ ░ ░ ░ ░░ ░ ░ ▒  ░      ░    ░    ░  ▒ ░   ░   ░  ░  ░  
  ░ ░        ░ ░         ░    ░       ░     ░  ░      ░  
░                                  ░                     

""")
    time.sleep(2)
    print("""

Sorry: You have been eaten alive by zombies. You are now one of them. 

""")
    print(40 * "-")
    time.sleep(2)
    play_again = input(f"""
{your_name} would you like to play again?
        a) Yes
        b) No
""").lower()
    
    ## start of conditional statement
    if "a" in play_again or "1" in play_again or "yes" in play_again:
        time.sleep(1)
        print("""
Great! You will start again at the hospital round.

- Maybe this time your zombie apocalypse skills are improved. -
""")
        ##take user to round 1
        hospital_setup()

    elif "b" in play_again or "2" in play_again or "no" in play_again:
        time.sleep(1)
        print("""
No problem! Enjoy your after-life!
""")
        exit()

##############################################################################
############## Alive Function ##############
## This function defines the win in the game! The user made it to their family alive.

def alive():
    print(40 * "-")
    time.sleep(2)
    print(f"""

Congratulations! You have stayed alive long enough to meet {your_SO} again!

The person in the radio is part of a group of survivors that are living in the outskirts of town.

They were on a mission in the city and will take you back with them were you will meet up with your family!

""")
    time.sleep(2)
    print(40 * "-")
    
    ## Asking if user wants to play again
    play_again = input(f"""
{your_name} would you like to play again?
        a) Yes
        b) No
""").lower()
    
    ## start of conditional statement
    if "a" in play_again or "1" in play_again or "yes" in play_again:
        time.sleep(1)
        print("""
Great! You will start again at round 1 at the hospital.
""")
        ##take user to round 1
        hospital_setup()

    elif "b" in play_again or "2" in play_again or "no" in play_again:
        time.sleep(1)
        print("""
No problem! Enjoy your post-apocalyptic life!


▒███████▒ ▒█████   ███▄ ▄███▓ ▄▄▄▄    ██▓▓█████   ██████ 
▒ ▒ ▒ ▄▀░▒██▒  ██▒▓██▒▀█▀ ██▒▓█████▄ ▓██▒▓█   ▀ ▒██    ▒ 
░ ▒ ▄▀▒░ ▒██░  ██▒▓██    ▓██░▒██▒ ▄██▒██▒▒███   ░ ▓██▄   
  ▄▀▒   ░▒██   ██░▒██    ▒██ ▒██░█▀  ░██░▒▓█  ▄   ▒   ██▒
▒███████▒░ ████▓▒░▒██▒   ░██▒░▓█  ▀█▓░██░░▒████▒▒██████▒▒
░▒▒ ▓░▒░▒░ ▒░▒░▒░ ░ ▒░   ░  ░░▒▓███▀▒░▓  ░░ ▒░ ░▒ ▒▓▒ ▒ ░
░░▒ ▒ ░ ▒  ░ ▒ ▒░ ░  ░      ░▒░▒   ░  ▒ ░ ░ ░  ░░ ░▒  ░ ░
░ ░ ░ ░ ░░ ░ ░ ▒  ░      ░    ░    ░  ▒ ░   ░   ░  ░  ░  
  ░ ░        ░ ░         ░    ░       ░     ░  ░      ░  
░                                  ░                     


""")
        exit()
  
        
##################### INITIALIZE GAME ########################################
setup()

```
