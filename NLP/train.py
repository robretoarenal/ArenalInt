from utils import SentimentTrain

def main():
    #This function will return features and labels cleaned.

    t = SentimentTrain("Data").train()
    print(t)

if __name__ == "__main__":
    main()