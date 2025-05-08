from collections import defaultdict, Counter
import math

class HiddenMarkovModel:
    def __init__(self):
        self.transition_probs = defaultdict(lambda: defaultdict(float))
        self.emission_probs = defaultdict(lambda: defaultdict(float))
        self.tags = set()
        self.vocab = set()
        self.tag_counts = Counter()
        self.start_token = "<S>"
        self.end_token = "<E>"

    def train(self, data, laplace=True):
        transition_counts = defaultdict(Counter)
        emission_counts = defaultdict(Counter)
        tag_counts = Counter()
        vocab = set()

        for sentence in data:
            prev_tag = self.start_token
            tag_counts[prev_tag] += 1  # Track start token count

            for word, tag in sentence:
                emission_counts[tag][word] += 1
                transition_counts[prev_tag][tag] += 1
                tag_counts[tag] += 1
                vocab.add(word)
                prev_tag = tag

            transition_counts[prev_tag][self.end_token] += 1  # End transition

        self.tags = set(tag_counts.keys()) - {self.start_token}
        self.vocab = vocab
        self.tag_counts = tag_counts

        V = len(vocab)
        T = len(self.tags)

        # --- Transition Probabilities ---
        for prev_tag in transition_counts:
            total = sum(transition_counts[prev_tag].values())
            for tag in self.tags.union({self.end_token}):
                count = transition_counts[prev_tag][tag]
                if laplace:
                    prob = (count + 1) / (total + T + 1)
                else:
                    prob = count / total if total > 0 else 0.0
                self.transition_probs[prev_tag][tag] = prob

        # --- Emission Probabilities ---
        for tag in emission_counts:
            total = sum(emission_counts[tag].values())
            for word in vocab:
                count = emission_counts[tag][word]
                if laplace:
                    prob = (count + 1) / (total + V)
                else:
                    prob = count / total if total > 0 else 0.0
                self.emission_probs[tag][word] = prob

    def get_transition_probs(self):
        return dict(self.transition_probs)

    def get_emission_probs(self):
        return dict(self.emission_probs)

    def predict(self, sentences):
        """
        Predict tags for each sentence using Viterbi Algorithm.
        Input: list of sentence strings (list[str])
        Output: list of list of predicted tags
        """
        results = []
        for sentence in sentences:
            words = sentence.strip().split()
            V = [{}]
            path = {}

            # Initialization step
            for tag in self.tags:
                emit_prob = self.emission_probs[tag].get(words[0], 1e-6)
                trans_prob = self.transition_probs[self.start_token].get(tag, 1e-6)
                V[0][tag] = math.log(trans_prob) + math.log(emit_prob)
                path[tag] = [tag]

            # Recursive step
            for t in range(1, len(words)):
                V.append({})
                new_path = {}

                for curr_tag in self.tags:
                    max_prob, best_prev = max(
                        ((V[t - 1][prev_tag] +
                          math.log(self.transition_probs[prev_tag].get(curr_tag, 1e-6)) +
                          math.log(self.emission_probs[curr_tag].get(words[t], 1e-6)), prev_tag)
                         for prev_tag in self.tags),
                        key=lambda x: x[0]
                    )
                    V[t][curr_tag] = max_prob
                    new_path[curr_tag] = path[best_prev] + [curr_tag]

                path = new_path

            # Termination
            n = len(words) - 1
            max_prob, final_tag = max((V[n][tag], tag) for tag in self.tags)
            results.append(path[final_tag])

        return results

    def print_transition_probs(self):
        for state, transitions in  dict(self.transition_probs).items():
            print(f"From state: {state}")
            for next_state, probability in transitions.items():
                print(f"  To {next_state}: {probability:.4f}")
        print()
    
    def print_emission_probs(self):
        for state, emissions in dict(self.emission_probs).items():
            print(f"From state: {state}")
            for word, probability in emissions.items():
                print(f"  Word '{word}': {probability:.4f}")
        print()