import pennylane as qml
import numpy as np

class QuantumPolicy:
    """
    Advanced Quantum Policy (v5.4.2) with Multi-Agent Decision Intelligence.
    Supports Influence Attribution, Narrative Risk Estimation, and Explainability.
    """

    def __init__(self, n_qubits=4):
        self.n_qubits = n_qubits
        self.dev = qml.device("default.mixed", wires=n_qubits)
        self.weights = np.random.uniform(0, np.pi, n_qubits)

        @qml.qnode(self.dev)
        def circuit(state_vec, weights, trait_vec=None, eps=0.0):
            if trait_vec is None:
                trait_vec = np.zeros(self.n_qubits)

            for i in range(self.n_qubits):
                # 1. State Encoding
                qml.RY(state_vec[i % len(state_vec)], wires=i)
                # 2. Character Trait Biasing
                qml.RY(trait_vec[i % len(trait_vec)] * np.pi, wires=i)
                # 3. Variational Layer
                qml.RZ(weights[i], wires=i)

            return qml.probs(wires=range(self.n_qubits))

        self.circuit = circuit

    def get_action_probs(self, state_vec, num_actions, trait_vec=None, eps=0.0):
        probs = self.circuit(state_vec, self.weights, trait_vec=trait_vec, eps=eps)
        if len(probs) > num_actions:
            probs = probs[:num_actions]
        probs = probs / (np.sum(probs) + 1e-10)
        return probs

    def update(self, reward, state_vec, action_idx, lr=0.01):
        self.weights += lr * reward * (action_idx / (self.n_qubits + 1))
        self.weights = np.clip(self.weights, 0, 2 * np.pi)

    def get_feature_sensitivity(self, state_vec, trait_vec=None, eps_noise=0.0):
        delta = 0.05
        base_probs = self.get_action_probs(state_vec, 2**self.n_qubits, trait_vec=trait_vec, eps=eps_noise)
        sensitivities = []
        for i in range(len(state_vec)):
            perturbed_state = state_vec.copy()
            perturbed_state[i] += delta
            perturbed_probs = self.get_action_probs(perturbed_state, 2**self.n_qubits, trait_vec=trait_vec, eps=eps_noise)
            sensitivity = np.sum(np.abs(perturbed_probs - base_probs)) / delta
            sensitivities.append(float(sensitivity))
        total = sum(sensitivities) + 1e-9
        return [s / total for s in sensitivities]

    def get_entropy(self, state_vec, trait_vec=None, eps_noise=0.0):
        probs = self.get_action_probs(state_vec, 2**self.n_qubits, trait_vec=trait_vec, eps=eps_noise)
        probs = np.clip(probs, 1e-10, 1.0)
        return float(-np.sum(probs * np.log2(probs)))

    def estimate_narrative_risk(self, state_vec, trait_vec=None, eps_noise=0.0):
        """Feature 5: Narrative Risk Assessment."""
        entropy = self.get_entropy(state_vec, trait_vec, eps_noise)
        tension = state_vec[0]
        risk = (entropy / self.n_qubits) * 0.6 + tension * 0.4
        return float(np.clip(risk, 0, 1))

    def get_influence_attribution(self, state_vec, trait_vec=None):
        """Feature 4: Agent Influence Attribution."""
        if trait_vec is None:
            return {"Protagonist": 0.33, "Antagonist": 0.33, "Ally": 0.34}

        base_probs = self.get_action_probs(state_vec, 2**self.n_qubits, trait_vec=trait_vec)
        influences = []
        for i in range(len(trait_vec)):
            perturbed_traits = trait_vec.copy()
            perturbed_traits[i] = np.clip(trait_vec[i] + 0.1, 0, 1)
            new_probs = self.get_action_probs(state_vec, 2**self.n_qubits, perturbed_traits)
            influences.append(np.sum(np.abs(new_probs - base_probs)))

        total = sum(influences) + 1e-9
        norm_inf = [i / total for i in influences]
        return {
            "Protagonist": float(norm_inf[0]),
            "Antagonist": float(norm_inf[1]),
            "Ally": float(norm_inf[2])
        }
