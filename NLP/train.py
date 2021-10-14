from utils import SentimentTrain

def main():
    """Training the model by using utils.SentimentTrain method. 
    """

    t = SentimentTrain("SentimentAnalysis/Data").train()

if __name__ == "__main__":
    main()