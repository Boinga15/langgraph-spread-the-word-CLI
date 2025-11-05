from agent_interface import headline_workflow

import os
import random

if __name__ == "__main__":
    os.system("cls")

    totalScore = 0
    headlines = []
    headlinesLeft = 20

    while headlinesLeft > 0:
        print(f"Current Score: {totalScore}")
        print(f"Headlines Left: {headlinesLeft}\n")
        
        print("Previous Headlines:")

        for headline in headlines:
            print(f"\"{headline[0]}\" - Success Rate: {headline[1]}%")
        
        print("\nEnter your next headline:")
        newHeadline = input("> ")

        os.system("cls")
        
        print("Processing headline, please stand by...")
        
        result = headline_workflow.invoke({"headline": newHeadline, "previous_headlines": headlines})
        
        # Normalize results to range.
        for resultKey in result.keys():
            if result[resultKey] < 0:
                result[resultKey] = 0
            elif result[resultKey] > 100:
                result[resultKey] = 100
        
        os.system("cls")

        contextScore = result.get("contextScore")
        noContextScore = result.get("noContextScore")
        interestScore = result.get("interestScore")
        
        print(f"Predicted Believability: {contextScore}%")
        print(f"Predicted Interest: {noContextScore}%")

        input("\nPress enter to see how well the headline did...")

        believability = random.choice(range(min(100, contextScore + round(interestScore * 0.5)), 101))
        finalScore = round(1000 * (believability / 100.0) * (2 - (noContextScore / 100.0)))

        totalScore += finalScore

        headlines.append([newHeadline, believability])
        headlinesLeft -= 1

        print(f"Headline Believability: {believability}%")
        print(f"Score: {finalScore} (Total Score: {totalScore})")

        input("\nPress enter to continue...")
    
        os.system("cls")

    print(f"Final Score: {totalScore}")
    input("\nPress enter to finish...")