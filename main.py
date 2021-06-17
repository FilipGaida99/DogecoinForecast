from sys import argv

from forecast.Forecaster import forecast

if __name__ == '__main__':
    days = 40
    if len(argv) > 1:
        days = argv[1]
    forecast('datasets/Dogecoin-original.csv', days, True)
