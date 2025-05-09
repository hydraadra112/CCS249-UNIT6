from collections import defaultdict, Counter
import math

class HMMTagger:
    def __init__(self):
        self.transition_probs = defaultdict(lambda: defaultdict(float))
        self.emission_probs = defaultdict(lambda: defaultdict(float))
        self.tag_counts = defaultdict(int)
        self.vocab = set()
        self.tags = set()
        self.laplace = True # Automated laplace

    def train(self, tagged_sentences):
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

            transition_counts[prev_tag]["<E>"] += 1
            tag_counts["<E>"] += 1
            self.tags.add("<E>")

        self.tag_counts = tag_counts

        for prev_tag in transition_counts:
            total = sum(transition_counts[prev_tag].values())
            for next_tag in self.tags:
                count = transition_counts[prev_tag][next_tag]
                prob = (count + 1) / (total + len(self.tags)) if self.laplace else count / total
                self.transition_probs[prev_tag][next_tag] = prob

        for tag in emission_counts:
            total = sum(emission_counts[tag].values())
            for word in self.vocab:
                count = emission_counts[tag][word]
                prob = (count + 1) / (total + len(self.vocab)) if self.laplace else count / total
                self.emission_probs[tag][word] = prob

    def viterbi(self, sentence):
        V = [{}]
        path = {}

        for tag in self.tags:
            if tag == "<E>": continue
            trans_p = self.transition_probs["<S>"].get(tag, 1e-6)
            emit_p = self.emission_probs[tag].get(sentence[0], 1e-6)
            V[0][tag] = math.log(trans_p) + math.log(emit_p)
            path[tag] = [tag]

        for t in range(1, len(sentence)):
            V.append({})
            new_path = {}

            for curr_tag in self.tags:
                if curr_tag == "<E>": continue
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

        (prob, best_final_tag) = max(
            ((V[-1][tag] + math.log(self.transition_probs[tag].get("<E>", 1e-6)), tag)
             for tag in V[-1]),
            key=lambda x: x[0]
        )

        return path[best_final_tag]
    
    def evaluate_known_sequence(self, sentence, tags):
        print(f"Evaluating known tag sequence:")
        prob = 1.0

        prev_tag = "<S>"
        for word, tag in zip(sentence, tags):
            trans_p = self.transition_probs[prev_tag].get(tag, 1e-6)
            emit_p = self.emission_probs[tag].get(word, 1e-6)
            step_prob = trans_p * emit_p
            print(f"P({tag} | {prev_tag}) = {trans_p:.6f}, P({word} | {tag}) = {emit_p:.6f} → Step = {step_prob:.8f}")
            prob *= step_prob
            prev_tag = tag

        end_p = self.transition_probs[prev_tag].get("<E>", 1e-6)
        print(f"P(<E> | {prev_tag}) = {end_p:.6f}")
        prob *= end_p

        print(f"\nFinal probability of the sequence: {prob:.12f}\n")    