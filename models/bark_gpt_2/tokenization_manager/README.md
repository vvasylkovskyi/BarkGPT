# Tokenization Manager

The tokenization manager takes a raw dataset `Dataset` object and converts it into ready for training dataset. Couple of transformations occur along the way:

1. The dataset is tokenized
2. The tokenized dataset text is grouped
3. The grouped text is collated (with padding)

Along the two initial steps, the results are cached, for fast retrieval for subsquent training steps.
