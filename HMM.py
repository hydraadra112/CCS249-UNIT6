from collections import defaultdict, Counter
import math

class HMMTagger:
    def __init__(self):
        self.transition_probs = defaultdict(lambda: defaultdict(float))
        self.emission_probs = defaultdict(lambda: defaultdict(float))
        self.tag_counts = defaultdict(int)
        self.vocab = set()
        self.tags = set()
        self.laplace = True

    def train(self, tagged_sentences):
        # Count occurrences
        transition_counts = defaultdict(Counter)
        emission_counts = defaultdict(Counter)
        tag_counts = Counter()

        for sentence in tagged_sentences:
            prev_tag = "<S>"
            tag_counts[prev_tag] += 1

            for word, tag in sentence:
                emission_counts[tag][word] += 1
                transition_counts[prev_tag][tag] += 1
                tag_counts[tag] += 1
                self.vocab.add(word)
                self.tags.add(tag)
                prev_tag = tag

            transition_counts[prev_tag]["<E>"] += 1  # end symbol
            tag_counts["<E>"] += 1
            self.tags.add("<E>")

        self.tag_counts = tag_counts

        for prev_tag in transition_counts:
            total = sum(transition_counts[prev_tag].values())
            for next_tag in self.tags:
                count = transition_counts[prev_tag][next_tag]
                if self.laplace:
                    prob = (count + 1) / (total + len(self.tags))
                else:
                    prob = count / total if total > 0 else 0
                self.transition_probs[prev_tag][next_tag] = prob

        for tag in emission_counts:
            total = sum(emission_counts[tag].values())
            for word in self.vocab:
                count = emission_counts[tag][word]
                if self.laplace:
                    prob = (count + 1) / (total + len(self.vocab))
                else:
                    prob = count / total if total > 0 else 0
                self.emission_probs[tag][word] = prob

    def set_laplace(self, use_laplace=True):
        self.laplace = use_laplace

    def viterbi(self, sentence):
        V = [{}]
        path = {}

        # Initialize base cases (t == 0)
        for tag in self.tags:
            if tag == "<E>":
                continue
            trans_p = self.transition_probs["<S>"].get(tag, 1e-6)
            emit_p = self.emission_probs[tag].get(sentence[0], 1e-6)
            V[0][tag] = math.log(trans_p) + math.log(emit_p)
            path[tag] = [tag]

        # Run Viterbi for t > 0
        for t in range(1, len(sentence)):
            V.append({})
            new_path = {}

            for curr_tag in self.tags:
                if curr_tag == "<E>":
                    continue
                (prob, best_prev_tag) = max(
                    (
                        (V[t - 1][prev_tag] + 
                         math.log(self.transition_probs[prev_tag].get(curr_tag, 1e-6)) +
                         math.log(self.emission_probs[curr_tag].get(sentence[t], 1e-6)),
                         prev_tag)
                        for prev_tag in V[t - 1]
                    ),
                    key=lambda x: x[0]
                )
                V[t][curr_tag] = prob
                new_path[curr_tag] = path[best_prev_tag] + [curr_tag]

            path = new_path

        # End transition
        (prob, best_final_tag) = max(
            ((V[-1][tag] + math.log(self.transition_probs[tag].get("<E>", 1e-6)), tag)
             for tag in V[-1]),
            key=lambda x: x[0]
        )

        return path[best_final_tag], prob