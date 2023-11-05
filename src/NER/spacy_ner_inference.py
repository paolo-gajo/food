from recipe_scrapers import scrape_me
import argparse

recipe_url_en = "https://www.allrecipes.com/recipe/246868/pecan-sour-cream-coffee-cake/"
recipe_url_it = "https://ricette.giallozafferano.it/Polpette-alla-cacciatora.html"
recipe_url_es_1 = "https://mahatmarice.com/es/recetas/autentica-paella-espanola-con-marisco/" # doesn't work because the recipe scraper only works with a fixed number of websites
recipe_url_es_2 = "https://www.comedera.com/como-hacer-paella-de-marisco/" # doesn't work because the recipe scraper only works with a fixed number of websites

def ingredient_scraper(recipe_url):
    scraper = scrape_me(recipe_url)
    ingredients = scraper.ingredients()
    return ingredients

from fractions import Fraction
import re

def fraction_to_mixed_number(fraction: Fraction) -> str:
  if fraction.numerator >= fraction.denominator:
    whole, remainder = divmod(fraction.numerator, fraction.denominator)
    if remainder == 0:
      return str(whole)
    else:
      return f"{whole} {Fraction(remainder, fraction.denominator)}"
  else:
    return str(fraction)


def convert_floats_to_fractions(text: str) -> str:
    return re.sub(
        r'\b-?\d+\.\d+\b',
        lambda match: fraction_to_mixed_number(
            Fraction(float(match.group())).limit_denominator()), text
        )


def process_text(text, model):
  """
  A wrapper function to pre-process text and run it through our pipeline.
  """
  return model(convert_floats_to_fractions(text))

def main():
   parser = argparse.ArgumentParser(description='Spacy NER Inference')
   parser.add_argument('recipe_url')
   args = parser.parse_args()
   ingredient_list = ingredient_scraper(args.recipe_url)
   print(ingredient_list)


if __name__ == '__main__':   
   main()