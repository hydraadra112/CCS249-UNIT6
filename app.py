from HMM import HiddenMarkovModel
# Training data: each word is (word, tag)
training_data = [
    [("The", "DET"), ("cat", "NOUN"), ("sleeps", "VERB")],
    [("A", "DET"), ("dog", "NOUN"), ("barks", "VERB")],
    [("The", "DET"), ("dog", "NOUN"), ("sleeps", "VERB")],
    [("My", "DET"), ("dog", "NOUN"), ("runs", "VERB"), ("fast", "ADV")],
    [("A", "DET"), ("cat", "NOUN"), ("meows", "VERB"), ("loudly", "ADV")],
    [("Your", "DET"), ("cat", "NOUN"), ("runs", "VERB")],
    [("The", "DET"), ("bird", "NOUN"), ("sings", "VERB"), ("sweetly", "ADV")],
    [("A", "DET"), ("bird", "NOUN"), ("chirps", "VERB")]
]

hmm = HiddenMarkovModel()
hmm.train(training_data)

# Test sentences
test_sentences = [
    ["The", "cat", "meows"],
    ["My", "dog", "barks", "loudly"]
]

for sentence in test_sentences:
    tags = hmm.viterbi(sentence)
    print(f"Sentence: {sentence}")
    print(f"Predicted Tags: {tags}")