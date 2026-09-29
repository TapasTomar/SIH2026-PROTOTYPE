# ============================================================
# SIH26055 SMART SCAN STRATEGY
# ONLINE ML ELECTRONIC SUPPORT SCHEDULER
# ============================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

BANDWIDTH = 100.0
DWELL_TIME = 2000.0

EPISODES = 30

ALPHA = 1.5
INITIAL_EPSILON = 0.25
MIN_EPSILON = 0.03

RIDGE = 1.0

HIT_REWARD = 10.0
MISS_PENALTY = -1.0
NEW_EMITTER_BONUS = 5.0
EARLY_INTERCEPT_BONUS = 3.0

FALSE_ALARM_PROBABILITY = 0.02
BASE_DETECTION_PROBABILITY = 0.70

DYNAMIC_ENVIRONMENT = True

# None = different run each time
RANDOM_SEED = None

OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SIMULATED RF ENVIRONMENT
# ============================================================

class SimulatedRFEnvironment:

    def __init__(self, dataframe, rng):

        self.original_data = dataframe.copy()
        self.rng = rng

        required_columns = [
            "ToA",
            "Frequency",
            "PulseWidth",
            "AoA",
            "Amplitude",
            "Label"
        ]

        for column in required_columns:
            if column not in dataframe.columns:
                raise ValueError(
                    f"Required column missing: {column}"
                )

        self.data = dataframe.copy()

        self.data = self.data.sort_values(
            "ToA"
        ).reset_index(drop=True)

        self.min_time = float(
            self.data["ToA"].min()
        )

        self.max_time = float(
            self.data["ToA"].max()
        )

        self.min_frequency = float(
            self.data["Frequency"].min()
        )

        self.max_frequency = float(
            self.data["Frequency"].max()
        )

        self.current_time = self.min_time

        self.episode_data = None


    # --------------------------------------------------------
    # Reset environment for a new episode
    # --------------------------------------------------------

    def reset(self):

        self.current_time = self.min_time

        self.episode_data = self.data.copy()

        if DYNAMIC_ENVIRONMENT:

            time_variation = self.rng.normal(
                0,
                20,
                len(self.episode_data)
            )

            frequency_variation = self.rng.normal(
                0,
                25,
                len(self.episode_data)
            )

            amplitude_variation = self.rng.normal(
                0,
                1.5,
                len(self.episode_data)
            )

            self.episode_data["ToA"] = (
                self.episode_data["ToA"]
                + time_variation
            )

            self.episode_data["Frequency"] = (
                self.episode_data["Frequency"]
                + frequency_variation
            )

            self.episode_data["Amplitude"] = (
                self.episode_data["Amplitude"]
                + amplitude_variation
            )

        self.episode_data = (
            self.episode_data
            .sort_values("ToA")
            .reset_index(drop=True)
        )


    # --------------------------------------------------------
    # Observe selected frequency
    # --------------------------------------------------------

    def observe(self, center_frequency):

        start_time = self.current_time

        end_time = (
            self.current_time
            + DWELL_TIME
        )

        self.current_time = end_time

        time_window = self.episode_data[
            (self.episode_data["ToA"] >= start_time)
            &
            (self.episode_data["ToA"] < end_time)
        ]

        frequency_window = time_window[
            abs(
                time_window["Frequency"]
                - center_frequency
            )
            <= BANDWIDTH / 2
        ]

        detected_events = []

        for _, event in frequency_window.iterrows():

            detection_probability = (
                BASE_DETECTION_PROBABILITY
            )

            if self.rng.random() < detection_probability:
                detected_events.append(event)

        false_alarm = (
            self.rng.random()
            < FALSE_ALARM_PROBABILITY
        )

        if false_alarm and len(detected_events) == 0:

            false_alarm_count = 1

        else:

            false_alarm_count = 0

        if len(detected_events) > 0:

            hit = True

        else:

            hit = False

        return {
            "hit": hit,
            "events": detected_events,
            "false_alarm": false_alarm_count,
            "actual_events": len(frequency_window),
            "detected_events": len(detected_events)
        }


# ============================================================
# CREATE FREQUENCY ACTIONS
# ============================================================

def create_frequency_actions(
    min_frequency,
    max_frequency
):

    actions = []

    frequency = (
        np.floor(min_frequency / BANDWIDTH)
        * BANDWIDTH
    )

    while frequency <= max_frequency:

        actions.append(float(frequency))

        frequency += BANDWIDTH

    return np.array(actions)


# ============================================================
# ONLINE CONTEXTUAL BANDIT
# ============================================================

class ContextualBanditScheduler:

    def __init__(
        self,
        actions,
        min_frequency,
        max_frequency
    ):

        self.actions = actions

        self.num_actions = len(actions)

        self.min_frequency = min_frequency
        self.max_frequency = max_frequency

        # Context:
        #
        # 1. bias
        # 2. normalized frequency
        # 3. normalized time
        # 4. historical hit rate
        # 5. recent result
        # 6. time since previous scan

        self.context_size = 6

        self.A = np.array([
            np.eye(self.context_size)
            * RIDGE
            for _ in range(self.num_actions)
        ])

        self.b = np.zeros(
            (
                self.num_actions,
                self.context_size
            )
        )

        self.counts = np.zeros(
            self.num_actions
        )

        self.hits = np.zeros(
            self.num_actions
        )

        self.rewards = np.zeros(
            self.num_actions
        )

        self.total_updates = 0

        self.total_reward = 0.0

        self.last_result = 0.0

        self.last_scan_time = 0.0

        self.rng = np.random.default_rng()


    # --------------------------------------------------------
    # Create context
    # --------------------------------------------------------

    def build_context(
        self,
        action_index,
        current_time,
        max_time
    ):

        frequency = self.actions[action_index]

        normalized_frequency = (
            frequency - self.min_frequency
        ) / (
            self.max_frequency
            - self.min_frequency
            + 1e-9
        )

        normalized_time = (
            current_time
            / (max_time + 1e-9)
        )

        total_scans = np.sum(
            self.counts
        )

        if total_scans > 0:

            global_hit_rate = (
                np.sum(self.hits)
                / total_scans
            )

        else:

            global_hit_rate = 0.0

        time_since_scan = (
            current_time
            - self.last_scan_time
        )

        time_since_scan = min(
            time_since_scan / 10000.0,
            1.0
        )

        context = np.array([
            1.0,
            normalized_frequency,
            normalized_time,
            global_hit_rate,
            self.last_result,
            time_since_scan
        ])

        return context


    # --------------------------------------------------------
    # Select action
    # --------------------------------------------------------

    def select_action(
        self,
        current_time,
        max_time,
        epsilon
    ):

        # Exploration

        if self.rng.random() < epsilon:

            return self.rng.integers(
                self.num_actions
            )

        scores = []

        for action in range(
            self.num_actions
        ):

            context = self.build_context(
                action,
                current_time,
                max_time
            )

            A_inverse = np.linalg.inv(
                self.A[action]
            )

            theta = (
                A_inverse
                @ self.b[action]
            )

            exploitation = (
                theta @ context
            )

            exploration = ALPHA * np.sqrt(
                context
                @ A_inverse
                @ context
            )

            score = (
                exploitation
                + exploration
            )

            scores.append(score)

        return int(
            np.argmax(scores)
        )


    # --------------------------------------------------------
    # Update model after hit/miss
    # --------------------------------------------------------

    def update(
        self,
        action_index,
        context,
        reward,
        hit
    ):

        self.A[action_index] += np.outer(
            context,
            context
        )

        self.b[action_index] += (
            reward * context
        )

        self.counts[action_index] += 1

        if hit:

            self.hits[action_index] += 1

        self.rewards[action_index] += reward

        self.total_reward += reward

        self.total_updates += 1

        self.last_result = (
            1.0 if hit else 0.0
        )

        self.last_scan_time += DWELL_TIME


    # --------------------------------------------------------
    # Learned value
    # --------------------------------------------------------

    def learned_value(
        self,
        action_index
    ):

        A_inverse = np.linalg.inv(
            self.A[action_index]
        )

        theta = (
            A_inverse
            @ self.b[action_index]
        )

        context = self.build_context(
            action_index,
            0,
            1
        )

        return float(
            theta @ context
        )


    # --------------------------------------------------------
    # Estimated hit probability
    # --------------------------------------------------------

    def estimated_hit_probability(
        self,
        action_index
    ):

        if self.counts[action_index] == 0:

            return 0.0

        return float(
            self.hits[action_index]
            / self.counts[action_index]
        )


# ============================================================
# REWARD FUNCTION
# ============================================================

def calculate_reward(
    hit,
    new_emitter,
    early_intercept
):

    if not hit:

        return MISS_PENALTY

    reward = HIT_REWARD

    if new_emitter:

        reward += NEW_EMITTER_BONUS

    if early_intercept:

        reward += EARLY_INTERCEPT_BONUS

    return reward


# ============================================================
# RUN ONE EPISODE
# ============================================================

def run_episode(
    environment,
    scheduler,
    episode_number
):

    environment.reset()

    max_time = (
        environment.max_time
        - environment.min_time
    )

    current_time = 0.0

    scans = 0
    detections = 0
    false_alarms = 0
    actual_events = 0

    total_reward = 0.0

    detected_labels = set()

    episode_records = []

    prediction_correct = 0

    prediction_total = 0

    intercept_errors = []


    # Exploration decreases over training

    epsilon = max(
        MIN_EPSILON,
        INITIAL_EPSILON
        * (
            1
            - episode_number / EPISODES
        )
    )


    while current_time < max_time:

        action = scheduler.select_action(
            current_time,
            max_time,
            epsilon
        )

        context = scheduler.build_context(
            action,
            current_time,
            max_time
        )

        selected_frequency = (
            scheduler.actions[action]
        )

        # Model prediction

        predicted_hit = (
            scheduler.estimated_hit_probability(
                action
            ) >= 0.5
        )

        observation = environment.observe(
            selected_frequency
        )

        hit = observation["hit"]

        actual_event_count = (
            observation["actual_events"]
        )

        detected_event_count = (
            observation["detected_events"]
        )

        false_alarm_count = (
            observation["false_alarm"]
        )

        # ----------------------------------------------------
        # Prediction accuracy
        # ----------------------------------------------------

        if predicted_hit == hit:

            prediction_correct += 1

        prediction_total += 1


        # ----------------------------------------------------
        # Detection statistics
        # ----------------------------------------------------

        actual_events += actual_event_count

        detections += detected_event_count

        false_alarms += false_alarm_count


        # ----------------------------------------------------
        # New emitter detection
        # ----------------------------------------------------

        new_emitter = False

        if hit:

            for event in observation["events"]:

                label = event["Label"]

                if label not in detected_labels:

                    detected_labels.add(label)

                    new_emitter = True


        # ----------------------------------------------------
        # Early intercept
        # ----------------------------------------------------

        early_intercept = (
            current_time
            < max_time * 0.5
        )


        # ----------------------------------------------------
        # Reward
        # ----------------------------------------------------

        reward = calculate_reward(
            hit,
            new_emitter,
            early_intercept
        )

        total_reward += reward


        # ----------------------------------------------------
        # Approximate intercept-time error
        # ----------------------------------------------------

        if hit and observation["events"]:

            event_times = [
                float(event["ToA"])
                for event in observation["events"]
            ]

            actual_time = min(event_times)

            estimated_time = (
                current_time
                + DWELL_TIME / 2
            )

            intercept_error = abs(
                actual_time
                - estimated_time
            )

            intercept_errors.append(
                intercept_error
            )


        # ----------------------------------------------------
        # Update ML model
        # ----------------------------------------------------

        scheduler.update(
            action,
            context,
            reward,
            hit
        )


        # ----------------------------------------------------
        # Log
        # ----------------------------------------------------

        episode_records.append({

            "episode": episode_number,

            "time": current_time,

            "frequency": selected_frequency,

            "hit": int(hit),

            "actual_events": actual_event_count,

            "detected_events": detected_event_count,

            "false_alarm": false_alarm_count,

            "reward": reward,

            "epsilon": epsilon

        })


        scans += 1

        current_time += DWELL_TIME


    # ========================================================
    # METRICS
    # ========================================================

    if actual_events > 0:

        pd_value = (
            detections
            / actual_events
        )

    else:

        pd_value = 0.0


    negative_scans = max(
        scans - actual_events,
        1
    )

    pfa = (
        false_alarms
        / negative_scans
    )


    sensitivity = pd_value


    if prediction_total > 0:

        prediction_accuracy = (
            prediction_correct
            / prediction_total
        )

    else:

        prediction_accuracy = 0.0


    if len(intercept_errors) > 0:

        average_intercept_error = (
            np.mean(intercept_errors)
        )

    else:

        average_intercept_error = 0.0


    intercept_rate = (
        detections / scans
        if scans > 0
        else 0.0
    )


    metrics = {

        "episode": episode_number,

        "scans": scans,

        "detections": detections,

        "actual_events": actual_events,

        "false_alarms": false_alarms,

        "Pd": pd_value,

        "Pfa": pfa,

        "sensitivity": sensitivity,

        "intercept_rate": intercept_rate,

        "prediction_accuracy":
            prediction_accuracy,

        "intercept_time_error":
            average_intercept_error,

        "reward": total_reward

    }


    return metrics, episode_records


# ============================================================
# PLOT LEARNING CURVE
# ============================================================

def plot_learning(
    episode_metrics
):

    episodes = episode_metrics["episode"]

    plt.figure(figsize=(9, 5))

    plt.plot(
        episodes,
        episode_metrics["Pd"],
        marker="o"
    )

    plt.xlabel("Episode")

    plt.ylabel("Probability of Detection")

    plt.title(
        "Online ML Learning - Detection Performance"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/learning_pd.png",
        dpi=200
    )

    plt.close()


# ============================================================
# PLOT REWARD
# ============================================================

def plot_rewards(
    episode_metrics
):

    plt.figure(figsize=(9, 5))

    plt.plot(
        episode_metrics["episode"],
        episode_metrics["reward"],
        marker="o"
    )

    plt.xlabel("Episode")

    plt.ylabel("Reward")

    plt.title(
        "Online ML Reward Learning"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/learning_rewards.png",
        dpi=200
    )

    plt.close()


# ============================================================
# PLOT PREDICTION ACCURACY
# ============================================================

def plot_prediction_accuracy(
    episode_metrics
):

    plt.figure(figsize=(9, 5))

    plt.plot(
        episode_metrics["episode"],
        episode_metrics[
            "prediction_accuracy"
        ],
        marker="o"
    )

    plt.xlabel("Episode")

    plt.ylabel("Prediction Accuracy")

    plt.title(
        "Hit/Miss Prediction Accuracy"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/prediction_accuracy.png",
        dpi=200
    )

    plt.close()


# ============================================================
# TIME-FREQUENCY SCAN MAP
# ============================================================

def plot_scan_map(
    scan_log
):

    plt.figure(figsize=(10, 6))

    hits = scan_log[
        scan_log["hit"] == 1
    ]

    misses = scan_log[
        scan_log["hit"] == 0
    ]

    plt.scatter(
        misses["time"],
        misses["frequency"],
        marker="x",
        label="Miss"
    )

    plt.scatter(
        hits["time"],
        hits["frequency"],
        marker="o",
        label="Hit"
    )

    plt.xlabel("Time")

    plt.ylabel("Frequency")

    plt.title(
        "ML Frequency-Time Scan Decisions"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/time_frequency_scan.png",
        dpi=200
    )

    plt.close()


# ============================================================
# FREQUENCY USAGE
# ============================================================

def plot_frequency_usage(
    scheduler
):

    frequencies = scheduler.actions

    counts = scheduler.counts

    plt.figure(figsize=(10, 5))

    plt.bar(
        frequencies,
        counts,
        width=BANDWIDTH * 0.7
    )

    plt.xlabel("Frequency")

    plt.ylabel("Number of Scans")

    plt.title(
        "Learned Frequency Usage"
    )

    plt.grid(
        axis="y"
    )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/frequency_usage.png",
        dpi=200
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("==========================================")
    print("SIH26055 SMART SCAN STRATEGY")
    print("ONLINE ML ELECTRONIC SUPPORT SCHEDULER")
    print("==========================================")
    print()


    # --------------------------------------------------------
    # Dataset path
    # --------------------------------------------------------

    if len(sys.argv) > 1:

        dataset_path = sys.argv[1]

    else:

        dataset_path = "dataset.csv"


    if not os.path.exists(dataset_path):

        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )


    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    dataframe = pd.read_csv(
        dataset_path
    )


    print(
        f"Dataset shape: {dataframe.shape}"
    )

    print(
        f"Columns: {list(dataframe.columns)}"
    )

    print(
        f"Frequency range: "
        f"{dataframe['Frequency'].min():.4f} "
        f"to "
        f"{dataframe['Frequency'].max():.4f}"
    )

    print(
        "Emitter labels:",
        sorted(
            dataframe["Label"]
            .unique()
            .tolist()
        )
    )

    print()


    # --------------------------------------------------------
    # Random generator
    # --------------------------------------------------------

    rng = np.random.default_rng(
        RANDOM_SEED
    )


    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    environment = SimulatedRFEnvironment(
        dataframe,
        rng
    )


    # --------------------------------------------------------
    # Frequency actions
    # --------------------------------------------------------

    actions = create_frequency_actions(
        environment.min_frequency,
        environment.max_frequency
    )


    # --------------------------------------------------------
    # ML scheduler
    # --------------------------------------------------------

    scheduler = ContextualBanditScheduler(
        actions,
        environment.min_frequency,
        environment.max_frequency
    )

    scheduler.rng = rng


    print(
        f"Receiver bandwidth: {BANDWIDTH}"
    )

    print(
        f"Dwell time: {DWELL_TIME}"
    )

    print(
        f"Frequency actions: {len(actions)}"
    )

    print(
        f"Learning episodes: {EPISODES}"
    )

    print()


    # ========================================================
    # TRAINING
    # ========================================================

    all_metrics = []

    all_scan_records = []


    for episode in range(
        1,
        EPISODES + 1
    ):

        metrics, records = run_episode(
            environment,
            scheduler,
            episode
        )

        all_metrics.append(
            metrics
        )

        all_scan_records.extend(
            records
        )


    # ========================================================
    # SAVE DATA
    # ========================================================

    episode_metrics = pd.DataFrame(
        all_metrics
    )

    scan_log = pd.DataFrame(
        all_scan_records
    )


    episode_metrics.to_csv(
        f"{OUTPUT_DIR}/episode_metrics.csv",
        index=False
    )

    scan_log.to_csv(
        f"{OUTPUT_DIR}/scan_log.csv",
        index=False
    )


    # ========================================================
    # FINAL METRICS
    # ========================================================

    total_scans = int(
        episode_metrics["scans"].sum()
    )

    total_detections = int(
        episode_metrics["detections"].sum()
    )

    total_false_alarms = int(
        episode_metrics["false_alarms"].sum()
    )

    total_actual_events = int(
        episode_metrics["actual_events"].sum()
    )


    if total_actual_events > 0:

        final_pd = (
            total_detections
            / total_actual_events
        )

    else:

        final_pd = 0.0


    negative_scans = max(
        total_scans
        - total_actual_events,
        1
    )

    final_pfa = (
        total_false_alarms
        / negative_scans
    )


    final_sensitivity = final_pd


    final_intercept_rate = (
        total_detections
        / total_scans
    )


    final_prediction_accuracy = (
        episode_metrics[
            "prediction_accuracy"
        ].mean()
    )


    final_intercept_error = (
        episode_metrics[
            "intercept_time_error"
        ].mean()
    )


    average_reward = (
        episode_metrics[
            "reward"
        ].mean()
    )


    total_reward = (
        episode_metrics[
            "reward"
        ].sum()
    )


    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("==========================================")
    print("FINAL PERFORMANCE")
    print("==========================================")

    print(
        f"Total scans                 : "
        f"{total_scans}"
    )

    print(
        f"Total detections            : "
        f"{total_detections}"
    )

    print(
        f"Total false alarms          : "
        f"{total_false_alarms}"
    )

    print(
        f"Probability of Detection    : "
        f"{final_pd:.4f}"
    )

    print(
        f"Probability of False Alarm  : "
        f"{final_pfa:.4f}"
    )

    print(
        f"Sensitivity                 : "
        f"{final_sensitivity:.4f}"
    )

    print(
        f"Average Intercept Rate      : "
        f"{final_intercept_rate:.4f}"
    )

    print(
        f"Prediction Accuracy         : "
        f"{final_prediction_accuracy:.4f}"
    )

    print(
        f"Average Intercept Time Error: "
        f"{final_intercept_error:.2f}"
    )

    print(
        f"Average Reward              : "
        f"{average_reward:.4f}"
    )

    print(
        f"Total Reward                : "
        f"{total_reward:.2f}"
    )


    # ========================================================
    # LEARNING COMPARISON
    # ========================================================

    first_episode = episode_metrics.iloc[0]

    last_episode = episode_metrics.iloc[-1]


    print()
    print("LEARNING COMPARISON")
    print("------------------------------------------")

    print(
        f"First episode Pd : "
        f"{first_episode['Pd']:.4f}"
    )

    print(
        f"Last episode Pd  : "
        f"{last_episode['Pd']:.4f}"
    )

    print(
        f"First reward     : "
        f"{first_episode['reward']:.2f}"
    )

    print(
        f"Last reward      : "
        f"{last_episode['reward']:.2f}"
    )

    print(
        f"Model updates    : "
        f"{scheduler.total_updates}"
    )


    # ========================================================
    # TOP LEARNED ACTIONS
    # ========================================================

    print()
    print("TOP LEARNED FREQUENCY ACTIONS")
    print("------------------------------------------")


    learned_values = []

    for action in range(
        scheduler.num_actions
    ):

        value = scheduler.learned_value(
            action
        )

        probability = (
            scheduler.estimated_hit_probability(
                action
            )
        )

        learned_values.append(
            (
                action,
                value,
                probability
            )
        )


    learned_values.sort(
        key=lambda x: x[1],
        reverse=True
    )


    for rank, (
        action,
        value,
        probability
    ) in enumerate(
        learned_values[:5],
        start=1
    ):

        print(
            f"{rank}. "
            f"Frequency="
            f"{scheduler.actions[action]:.2f} | "
            f"ML value="
            f"{value:.4f} | "
            f"Hit probability="
            f"{probability:.3f} | "
            f"Scans="
            f"{int(scheduler.counts[action])} | "
            f"Hits="
            f"{int(scheduler.hits[action])}"
        )


    # ========================================================
    # PLOTS
    # ========================================================

    plot_learning(
        episode_metrics
    )

    plot_rewards(
        episode_metrics
    )

    plot_prediction_accuracy(
        episode_metrics
    )

    plot_scan_map(
        scan_log
    )

    plot_frequency_usage(
        scheduler
    )


    # ========================================================
    # SAVE LEARNED MODEL
    # ========================================================

    np.savez(
        f"{OUTPUT_DIR}/learned_model.npz",

        actions=scheduler.actions,

        A=scheduler.A,

        b=scheduler.b,

        counts=scheduler.counts,

        hits=scheduler.hits,

        rewards=scheduler.rewards

    )


    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("==========================================")
    print("PROTOTYPE COMPLETED")
    print("==========================================")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()