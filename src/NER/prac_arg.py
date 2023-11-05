
import argparse

def myfunction(input):
   print(input)

def main():
   parser = argparse.ArgumentParser(description='Spacy NER Inference')
   parser.add_argument('test1')
   args = parser.parse_args()
   myfunction(args.test1)

if __name__ == '__main__':   
   main()