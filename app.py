from HMM import HiddenMarkovModel
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
hmm.train(training_data, laplace=True)

# Print probabilities of trained data
print(f"TRANSITION PROBABILITIES:")
hmm.print_transition_probs()

print(f"\nEMISSION PROBABILITIES:")
hmm.print_emission_probs()

# Predict new tags
predicted = hmm.predict(["My bird sings sweetly", "The cat barks"])
print(predicted)