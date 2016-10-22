# imports
import os



class GroundTruths:

    def __init__(self, venues, trainingV, validationV, trainingD, validationD):
        self.venues = venues
        self.training_venue = trainingV
        self.validation_venue = validationV
        self.training_desc = trainingD
        self.validation_desc = validationD


def makeDict(venue_name_path):
    
    f = open(venue_name_path, 'r')
    d = dict()

    for line in f:
        a = line.split("\t")
        d[a[0]] = a[1].strip("\n")

    return d


def makeGroundTruths():

    dataset_dir = os.path.dirname(__file__)+"/CS2108-Vine-Dataset/"
    
    filename = "venue-name.txt"
    venues = makeDict(dataset_dir+filename)
    
    filename = "vine-venue-training.txt"
    training_venue = makeDict(dataset_dir+filename)
    
    filename = "vine-venue-validation.txt"
    validation_venue = makeDict(dataset_dir+filename)

    filename = "vine-desc-training.txt"
    training_desc = makeDict(dataset_dir+filename)

    filename = "vine-desc-validation.txt"
    validation_desc = makeDict(dataset_dir+filename)

    return GroundTruths(venues, training_venue, validation_venue, training_desc, validation_desc)


def getGroundTruths():
    
    return gt


gt = makeGroundTruths()


if __name__ == "__main__":

    makeGroundTruths()
    
