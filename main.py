from sys import argv

from forecast.Forecaster import forecast

if __name__ == '__main__':
    days = 10
    test = False
    if len(argv) > 1:
        days = argv[1]
    if len(argv) > 2:
        test = argv[2] == '-test'
    forecast('datasets/Dogecoin-original.csv', days, test)
