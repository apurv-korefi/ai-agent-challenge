import pandas as pd
import custom_parsers.icici_parser as parser

df = parser.parse("data/icici/icici_sample.pdf")
print(df.head())
