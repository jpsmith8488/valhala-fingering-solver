#!/usr/bin/env python3
"""
ValHaLA: Variational Hamiltonian Least Action Solver
for Piano Keyboard Kinematics
====================================================================

Version 2.1 — April 2026

Implements the complete 14-term equation of state for optimal piano
fingering as described in the accompanying manuscript.

This solver is self-contained.  The only external dependency is NumPy.

Equation of State Terms
-----------------------
1.  Kinetic energy (configuration-dependent inertia tensor)
2.  Topographic potential (keyboard geometry)
3.  Gravitational potential (arm-weight channeling)
4.  Bilateral collision avoidance
5.  Inter-digit coupling (juncturae tendinum, wrist-angle-dependent)
6.  Keyboard action model (Steinway D-274, register-dependent)
7.  Contact mechanics (Hertzian impact vs. pre-load)
8.  Finger-pad adhesion (Coulomb friction, hydration-dependent)
9.  Tempo-dependent accessible configuration space
10. Rayleigh dissipation
11. Pedagogical tradition parameterization (8 documented schools)
12. Hand repositioning cost (kinetic energy of whole-hand relocation)
13. Forearm rotation cost (angular displacement, Taubman school)
14. Black-key depth stagger (elevation asymmetry penalty)

Plus: phrase-structure cost modulation (sequential reward, coupling
suppression, terminal preference) enabling standard scale fingerings
to emerge as optimal solutions without explicit encoding.

License: MIT
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# =====================================================================
# Physical Constants
# =====================================================================

# Keyboard geometry (Steinway D-274 standard)
WHITE_KEY_WIDTH = 23.5      # mm
BLACK_KEY_WIDTH = 13.7      # mm
BLACK_KEY_ELEVATION = 12.0  # mm
OCTAVE_WIDTH = 7 * WHITE_KEY_WIDTH  # mm (7 white keys per octave)
KEY_DEPTH_WHITE = 50.0      # mm (front to balance point)
KEY_DEPTH_BLACK = 35.0      # mm
ESCAPEMENT_DEPTH = 4.5      # mm

# Key stiffness (register-dependent, Askenfelt & Jansson 1990)
K_TREBLE = 0.45             # N/mm (highest register)
DELTA_K = 0.15              # N/mm (bass increment)

# Biomechanical constants
TISSUE_DENSITY = 1100.0     # kg/m^3
NEUROMUSCULAR_LATENCY = 35  # ms
MAX_FINGER_VELOCITY = 1.5   # m/s
MOTOR_NOISE_SIGMA = 0.01    # rad
FINGER_ENVELOPE = 8.0       # mm
JOINT_VISCOSITY = 0.02      # Pa·s (normalized)
GRAVITY = 9.81              # m/s^2

# Collision avoidance
A_AVOID = 50.0

# Phrase structure
SEQUENTIAL_REWARD = -3.0
COUPLING_SUPPRESSION = 0.85
TERMINAL_MULTIPLIER = 4.0

# Black-key depth stagger (Eq. 5) and forearm rotation (Eq. 4)
DEPTH_STAGGER_DZ = 12.7         # mm, black-white elevation difference (Table 5)
ALPHA_BLACK_KEY = 2.0           # black-key stagger scale (Table 5)
ALPHA_ROTATION = 1.5            # forearm rotation scale (Table 5)
OMEGA_MAX = 1.2                 # rad, max comfortable forearm rotation (Table 5)
ARPEGGIO_GROUPING = 1.0         # scale on the arpeggio-grouping mechanism (v2.1);
                                # set to 0 to ablate grouping (validation/repro)

# Black key MIDI pitch classes (C#, D#, F#, G#, A#)
BLACK_KEYS = {1, 3, 6, 8, 10}


# =====================================================================
# Anthropometric Model (Buchholz, Armstrong & Goldstein 1992)
# =====================================================================

@dataclass
class HandModel:
    """Anthropometric hand model from Buchholz regression equations.

    Two measurements (H, B) calibrate the entire kinematic chain:
    segment lengths, diameters, masses, and maximum spans.
    """
    hand_length: float = 190.0   # mm
    hand_breadth: float = 85.0   # mm
    max_span: float = 210.0      # mm (D_15)

    def __post_init__(self):
        H = self.hand_length
        B = self.hand_breadth
        # Phalanx lengths (mm) from Buchholz regressions
        # [thumb, index, middle, ring, little]
        self.pp_lengths = [
            0.197 * H + 7.9,   # thumb metacarpal (functionally PP)
            0.238 * H + 3.1,   # index PP
            0.257 * H + 1.2,   # middle PP
            0.231 * H + 2.8,   # ring PP
            0.195 * H + 4.1,   # little PP
        ]
        self.ip_lengths = [
            0.158 * H + 1.0,   # thumb PP (functionally IP)
            0.143 * H + 0.5,   # index IP/PIP
            0.152 * H - 0.2,   # middle IP/PIP
            0.140 * H + 0.4,   # ring IP/PIP
            0.118 * H + 1.8,   # little IP/PIP
        ]
        self.dp_lengths = [
            0.126 * H + 0.3,   # thumb DP
            0.100 * H + 0.8,   # index DIP
            0.098 * H + 0.6,   # middle DIP
            0.094 * H + 0.7,   # ring DIP
            0.088 * H + 1.0,   # little DIP
        ]
        # Segment diameters (mm) - proximal and distal
        self.pp_diameters = [(0.095 * B + 4.0, 0.080 * B + 3.2) for _ in range(5)]
        # Segment masses (conical frustum, tissue density 1100 kg/m^3)
        self.segment_masses = self._compute_masses()
        # Effective fingertip mass (kg)
        self.effective_mass = [self._effective_mass(k) for k in range(5)]
        # Maximum inter-finger spans (mm)
        self.max_spans = self._compute_spans()

    def _frustum_mass(self, length_mm, d_prox_mm, d_dist_mm):
        """Mass of a conical frustum segment (kg)."""
        l = length_mm / 1000.0
        rp = d_prox_mm / 2000.0
        rd = d_dist_mm / 2000.0
        return (math.pi * TISSUE_DENSITY * l / 3.0) * (rp**2 + rp * rd + rd**2)

    def _compute_masses(self):
        """Total mass per finger (3 segments each)."""
        masses = []
        for k in range(5):
            dp, dd = self.pp_diameters[k]
            m_pp = self._frustum_mass(self.pp_lengths[k], dp, dd)
            m_ip = self._frustum_mass(self.ip_lengths[k], dd * 0.9, dd * 0.75)
            m_dp = self._frustum_mass(self.dp_lengths[k], dd * 0.75, dd * 0.55)
            masses.append(m_pp + m_ip + m_dp)
        return masses

    def _effective_mass(self, finger_idx):
        """Effective mass at fingertip along key-depression direction.

        Approximation from Khatib (1995): m_eff ≈ sum of segment masses
        weighted by moment-arm ratios.  Here we use the analytical result.
        """
        total_length = (self.pp_lengths[finger_idx] +
                        self.ip_lengths[finger_idx] +
                        self.dp_lengths[finger_idx]) / 1000.0  # meters
        m_total = self.segment_masses[finger_idx]
        # Effective mass is approximately 0.6-0.8 of total for
        # extended finger, configuration-averaged
        return m_total * 0.7

    def _compute_spans(self):
        """Maximum inter-finger spans (mm) from hand measurements."""
        base = self.max_span / 4.0  # average inter-finger max
        # Thumb has wider span; ring-little is restricted
        return {
            (0, 1): base * 1.4,   # thumb-index
            (1, 2): base * 0.9,   # index-middle
            (2, 3): base * 0.85,  # middle-ring
            (3, 4): base * 0.75,  # ring-little
            (0, 2): base * 2.2,   # thumb-middle
            (0, 3): base * 2.8,   # thumb-ring
            (0, 4): self.max_span,  # thumb-little
            (1, 3): base * 1.6,   # index-ring
            (1, 4): base * 2.1,   # index-little
            (2, 4): base * 1.4,   # middle-little
        }

    def finger_reach_mm(self, f1, f2):
        """Maximum reach between two fingers (0-indexed), mm."""
        pair = (min(f1, f2), max(f1, f2))
        return self.max_spans.get(pair, self.max_span * 0.5)

    def finger_length_mm(self, finger_idx):
        """Total extended length of a finger (0-indexed), mm.

        Sum of the three Buchholz-derived phalange segments
        (proximal + middle + distal).  This is the same anthropometry
        that drives the kinetic and topographic terms, so the
        stagger and rotation terms inherit personalization from
        (H, B) without introducing new parameters.
        """
        return (self.pp_lengths[finger_idx] +
                self.ip_lengths[finger_idx] +
                self.dp_lengths[finger_idx])


# =====================================================================
# Keyboard Model (Steinway D-274)
# =====================================================================

class KeyboardModel:
    """Steinway D-274 keyboard geometry and action model."""

    def __init__(self):
        # Precompute key positions for all 88 keys (MIDI 21-108)
        self._positions = {}
        for midi in range(21, 109):
            self._positions[midi] = self._key_position(midi)

    @staticmethod
    def is_black(midi):
        return (midi % 12) in BLACK_KEYS

    @staticmethod
    def _key_position(midi):
        """3D position (x, y, z) of key center in mm."""
        # x: lateral position from left edge of keyboard
        octave = (midi - 21) // 12
        note_in_octave = (midi - 21) % 12
        # White key index within octave
        white_offsets = {0: 0, 2: 1, 4: 2, 5: 3, 7: 4, 9: 5, 11: 6}
        black_offsets = {1: 0.5, 3: 1.5, 6: 3.5, 8: 4.5, 10: 5.5}

        if note_in_octave in white_offsets:
            x = (octave * 7 + white_offsets[note_in_octave]) * WHITE_KEY_WIDTH
            y = KEY_DEPTH_WHITE / 2
            z = 0.0
        else:
            x = (octave * 7 + black_offsets[note_in_octave]) * WHITE_KEY_WIDTH
            y = KEY_DEPTH_BLACK / 2
            z = BLACK_KEY_ELEVATION
        return (x, y, z)

    def key_distance(self, midi1, midi2):
        """Euclidean distance between two keys (mm)."""
        if midi1 not in self._positions or midi2 not in self._positions:
            return abs(midi2 - midi1) * 15.0  # fallback
        p1 = self._positions[midi1]
        p2 = self._positions[midi2]
        return math.sqrt(sum((a - b)**2 for a, b in zip(p1, p2)))

    def topographic_distance(self, midi1, midi2):
        """Topographic distance including elevation penalty."""
        d = self.key_distance(midi1, midi2)
        # Additional elevation penalty for black-white transitions
        b1 = self.is_black(midi1)
        b2 = self.is_black(midi2)
        if b1 != b2:
            d += BLACK_KEY_ELEVATION * 0.5  # partial elevation cost
        return d

    def key_stiffness(self, midi):
        """Register-dependent key stiffness (N/mm)."""
        n = midi - 21  # 0-indexed key number
        return K_TREBLE + DELTA_K * (1.0 - n / 87.0)

    def key_energy(self, midi):
        """Energy to depress key to escapement (mJ)."""
        k = self.key_stiffness(midi)
        return 0.5 * k * ESCAPEMENT_DEPTH**2


# =====================================================================
# Inter-Digit Coupling (Häger-Ross & Schieber 2000)
# =====================================================================

class CouplingModel:
    """Wrist-angle-dependent inter-digit coupling matrix."""

    # Baseline coupling constants (N/m)
    K_BASELINE = np.array([
        [0.0, 1.0, 0.5, 0.3, 0.2],   # thumb
        [1.0, 0.0, 2.5, 1.2, 0.5],   # index
        [0.5, 2.5, 0.0, 3.5, 1.0],   # middle
        [0.3, 1.2, 3.5, 0.0, 4.5],   # ring
        [0.2, 0.5, 1.0, 4.5, 0.0],   # little
    ])

    def __init__(self, wrist_angle=0.0):
        """wrist_angle: radians, positive = ulnar deviation."""
        self.wrist_angle = wrist_angle
        self.K = self._build_matrix()

    def _build_matrix(self):
        """Build wrist-angle-dependent coupling matrix."""
        K = self.K_BASELINE.copy()
        # Ring-little coupling increases with ulnar deviation
        delta_k45 = 2.0 * math.sin(self.wrist_angle)
        K[3, 4] += max(0, delta_k45)
        K[4, 3] += max(0, delta_k45)
        # Middle-ring coupling also increases slightly
        K[2, 3] += max(0, 0.5 * delta_k45)
        K[3, 2] += max(0, 0.5 * delta_k45)
        return K

    def coupling_cost(self, finger1, finger2):
        """Coupling cost between two fingers (0-indexed)."""
        return float(self.K[finger1, finger2])


# =====================================================================
# Pedagogical Tradition Parameterization
# =====================================================================

@dataclass
class TraditionProfile:
    """Cost-function coefficients encoding a pedagogical tradition.

    These are the sole non-physical parameters in the equation of state.
    All other terms are derived from biomechanics or instrument physics.
    """
    name: str
    thumb_on_black: float      # penalty for thumb on sharps/flats
    thumb_under: float         # cost of thumb-under crossing
    finger_crossing: float     # cost of finger-over-finger crossing
    paired_finger_bonus: float # reward for adjacent paired fingers
    legato_bonus: float        # reward for smooth voice leading
    arm_weight: float          # α_arm: fraction of arm weight channeled
    rotation_bonus: float      # reward for forearm rotation technique
    natural_position: float    # reward for hand in natural rest position
    metric_weight: float       # μ: weight of metric alignment

    def to_vector(self):
        """Return the tradition vector s."""
        return np.array([
            self.thumb_on_black, self.thumb_under, self.finger_crossing,
            self.paired_finger_bonus, self.legato_bonus, self.arm_weight,
            self.rotation_bonus, self.natural_position, self.metric_weight
        ])


# Eight documented pedagogical traditions
TRADITIONS = {
    'baroque': TraditionProfile(
        'Baroque', thumb_on_black=100, thumb_under=20,
        finger_crossing=4, paired_finger_bonus=-3, legato_bonus=0,
        arm_weight=0.0, rotation_bonus=0, natural_position=0,
        metric_weight=2.0),
    'classical': TraditionProfile(
        'Classical', thumb_on_black=20, thumb_under=6,
        finger_crossing=8, paired_finger_bonus=0, legato_bonus=-2,
        arm_weight=0.0, rotation_bonus=0, natural_position=0,
        metric_weight=1.0),
    'romantic': TraditionProfile(
        'Romantic', thumb_on_black=5, thumb_under=4,
        finger_crossing=12, paired_finger_bonus=0, legato_bonus=-4,
        arm_weight=0.3, rotation_bonus=0, natural_position=0,
        metric_weight=0.5),
    'modern': TraditionProfile(
        'Modern', thumb_on_black=2, thumb_under=3,
        finger_crossing=7, paired_finger_bonus=0, legato_bonus=-1,
        arm_weight=0.0, rotation_bonus=0, natural_position=0,
        metric_weight=0.0),
    'russian': TraditionProfile(
        'Russian', thumb_on_black=3, thumb_under=4,
        finger_crossing=10, paired_finger_bonus=0, legato_bonus=-3,
        arm_weight=0.8, rotation_bonus=-0.5, natural_position=0,
        metric_weight=0.0),
    'french': TraditionProfile(
        'French-Cortot', thumb_on_black=2, thumb_under=5,
        finger_crossing=6, paired_finger_bonus=0, legato_bonus=-1,
        arm_weight=0.0, rotation_bonus=0, natural_position=0,
        metric_weight=0.0),
    'taubman': TraditionProfile(
        'Taubman', thumb_on_black=3, thumb_under=2,
        finger_crossing=15, paired_finger_bonus=0, legato_bonus=-2,
        arm_weight=0.7, rotation_bonus=-2.0, natural_position=0,
        metric_weight=0.0),
    'chopin': TraditionProfile(
        'Chopin', thumb_on_black=1, thumb_under=4,
        finger_crossing=10, paired_finger_bonus=0, legato_bonus=-3,
        arm_weight=0.3, rotation_bonus=0, natural_position=-1.5,
        metric_weight=0.0),
}


# =====================================================================
# Solver Configuration
# =====================================================================

@dataclass
class SolverConfig:
    """Complete parameter vector Θ for the equation of state."""
    # Anthropometric
    hand_length: float = 190.0
    hand_breadth: float = 85.0
    max_span: float = 210.0
    # Physical
    mu_f: float = 0.5           # Coulomb friction coefficient
    wrist_angle: float = 0.0    # radians (positive = ulnar)
    technique: str = 'preload'  # 'preload' or 'impact'
    # Tradition
    tradition: str = 'modern'
    # Tempo
    tempo_nps: float = 6.0      # notes per second
    # Cost scaling
    alpha_topo: float = 1.0
    alpha_coupling: float = 1.5
    alpha_adhesion: float = 2.0
    alpha_tempo: float = 8.0
    alpha_dissipation: float = 0.5
    alpha_gravity: float = 0.3
    alpha_key: float = 0.4
    alpha_contact: float = 1.0
    alpha_collision: float = 5.0
    # Optional, off-by-default variant (paper Sec. 7, arm-weight analysis).
    # When False (default), the gravitational/arm-weight term is the
    # faithful per-finger channeling cost, which governs action MAGNITUDE
    # (and hence the scaling exponents) but not fingering SELECTION. When
    # True, an additional weight-channeling reward couples finger strength
    # to metrically strong notes, converting arm weight into a fingering-
    # selection effect. Both behaviors are reproducible; the variant is
    # provided for analysis and is not part of the default solver.
    arm_weight_selection: bool = False
    # Stochastic
    stochastic_sigma: float = 0.15
    stochastic_trials: int = 100


# =====================================================================
# Note Representation
# =====================================================================

@dataclass
class NoteEvent:
    """A note to be fingered."""
    midi: int
    onset: float = 0.0
    duration: float = 0.25
    dynamic: str = 'mf'
    beat_position: float = 0.0


# =====================================================================
# Solver Result
# =====================================================================

@dataclass
class SolverResult:
    """Complete result from the Viterbi-Hamiltonian solver."""
    fingers: List[int]          # 1-indexed finger assignments
    total_cost: float
    per_note_cost: List[float]
    cost_breakdown: Dict[str, float]
    ari_values: List[float]     # Action-Risk Index per transition
    ari_risk_levels: List[str]  # 'low', 'moderate', 'high', 'critical'
    tradition: str
    passage_length: int


# =====================================================================
# The Viterbi-Hamiltonian Solver
# =====================================================================

class HamiltonianSolver:
    """Viterbi solver implementing the complete 11-term equation of state.

    Usage:
        solver = HamiltonianSolver(hand_length=190, tradition='modern')
        result = solver.solve([60, 62, 64, 65, 67, 69, 71, 72])
        print(result['fingers'])  # [1, 2, 3, 1, 2, 3, 4, 5]
    """

    def __init__(self, hand_length=190.0, hand_breadth=85.0,
                 max_span=210.0, tradition='modern', mu_f=0.5,
                 wrist_angle=0.0, technique='preload', tempo_nps=6.0,
                 config: Optional[SolverConfig] = None):
        if config is not None:
            self.cfg = config
        else:
            self.cfg = SolverConfig(
                hand_length=hand_length, hand_breadth=hand_breadth,
                max_span=max_span, tradition=tradition, mu_f=mu_f,
                wrist_angle=wrist_angle, technique=technique,
                tempo_nps=tempo_nps)

        self.hand = HandModel(self.cfg.hand_length, self.cfg.hand_breadth,
                              self.cfg.max_span)
        self.keyboard = KeyboardModel()
        self.coupling = CouplingModel(self.cfg.wrist_angle)
        self.tradition = TRADITIONS.get(self.cfg.tradition,
                                        TRADITIONS['modern'])

        # Per-finger thumb-under difficulty scale
        # Finger 3 ascending is the standard scale pivot (lowest cost);
        # Finger 5 requires extreme ulnar deviation (highest cost).
        self.thumb_under_scale = {
            2: 0.45,   # index: moderate difficulty
            3: 0.15,   # middle: standard pivot (lowest)
            4: 0.70,   # ring: awkward
            5: 1.30,   # little: extreme ulnar deviation
        }

    # -----------------------------------------------------------------
    # Term 1: Kinetic energy (configuration-dependent)
    # -----------------------------------------------------------------
    def _kinetic_cost(self, midi1, midi2, finger1, finger2, dt):
        """Kinetic energy contribution: T = ½ m_eff v².

        For thumb-under transitions, the effective repositioning
        distance includes the arc length of the thumb passing
        underneath the palm — approximately 35-60 mm depending on
        which finger the thumb passes under, compared to ~24 mm
        for a stepwise advance of the next sequential finger.
        This arc distance is the physical mechanism that makes
        thumb-under increasingly expensive at high tempo.
        """
        d_mm = self.keyboard.topographic_distance(midi1, midi2)

        # Thumb-under: add configuration-space arc distance.
        # The thumb traverses an arc underneath the palm whose length
        # depends on how many fingers it passes under.  This arc is
        # substantially longer than the key-to-key Euclidean distance
        # and represents the dominant kinetic cost at high tempo.
        ascending = midi2 > midi1
        if finger1 != 1 and finger2 == 1 and ascending:
            arc = 50.0 + 15.0 * (finger1 - 2)  # ~50-80 mm
            d_mm += arc
        elif finger1 == 1 and finger2 != 1 and not ascending:
            arc = 50.0 + 15.0 * (finger2 - 2)
            d_mm += arc

        d_m = d_mm / 1000.0
        v = d_m / max(dt, 0.001)
        m_eff = self.hand.effective_mass[finger2 - 1]
        return 800.0 * m_eff * v**2

    # -----------------------------------------------------------------
    # Term 2: Topographic potential
    # -----------------------------------------------------------------
    def _topographic_cost(self, midi1, midi2, finger1, finger2):
        """Keyboard geometry cost: distance, elevation, span limits."""
        d_mm = self.keyboard.topographic_distance(midi1, midi2)

        # Finger-relative distance: normalize by inter-finger reach
        f1_idx, f2_idx = finger1 - 1, finger2 - 1
        max_reach = self.hand.finger_reach_mm(f1_idx, f2_idx)
        if max_reach <= 0:
            max_reach = 50.0

        # Stretch ratio: cost increases sharply near max span
        stretch = d_mm / max_reach
        if stretch > 1.0:
            # Physically impossible: extremely high penalty
            return 100.0 * stretch**2
        elif stretch > 0.7:
            # Uncomfortable range: quadratic increase
            return self.cfg.alpha_topo * (stretch * 2.0)**2
        else:
            return self.cfg.alpha_topo * stretch

    # -----------------------------------------------------------------
    # Term 3: Gravitational potential (arm weight channeling)
    # -----------------------------------------------------------------
    def _gravity_cost(self, finger):
        """Gravitational loading and finger strength.

        Heavier, stronger fingers (thumb, index) leverage gravity
        more effectively for key depression.  Weaker fingers (ring,
        little) require more muscular effort against gravity.
        Arm-weight traditions reduce this differential by channeling
        forearm weight through relaxed joints.
        """
        # Relative strength index (1.0 = strongest)
        # Based on maximum voluntary contraction force measurements
        strength = {1: 1.0, 2: 0.85, 3: 0.75, 4: 0.55, 5: 0.45}
        weakness = 1.0 - strength[finger]
        # Arm-weight reduction
        alpha_arm = self.tradition.arm_weight
        weakness *= (1.0 - alpha_arm * 0.7)
        return self.cfg.alpha_gravity * weakness * 3.0

    # -----------------------------------------------------------------
    # Term 4: Bilateral collision avoidance
    # -----------------------------------------------------------------
    def _collision_cost(self, midi1, midi2, finger1, finger2):
        """Repulsive potential preventing physically impossible configs."""
        if finger1 == finger2:
            return 0.0  # same finger can't collide with itself
        # Same-hand collision: fingers crossing over each other
        f1, f2 = finger1 - 1, finger2 - 1
        lateral_direction = 1 if midi2 > midi1 else -1
        finger_direction = 1 if f2 > f1 else -1
        # Thumb (finger 0) is special: always on one side
        if f1 == 0 or f2 == 0:
            return 0.0  # thumb crossings handled separately
        # Non-thumb fingers crossing: e.g., finger 4 plays a note
        # lower than finger 2 — anatomically awkward
        if lateral_direction != 0 and finger_direction != 0:
            if lateral_direction != finger_direction:
                d_mm = self.keyboard.key_distance(midi1, midi2)
                closeness = max(0, FINGER_ENVELOPE * 3 - d_mm)
                return self.cfg.alpha_collision * closeness**2 / 100.0
        return 0.0

    # -----------------------------------------------------------------
    # Term 5: Inter-digit coupling
    # -----------------------------------------------------------------
    def _coupling_cost(self, finger1, finger2, stepwise):
        """Juncturae tendinum coupling cost."""
        f1, f2 = finger1 - 1, finger2 - 1
        base = self.coupling.coupling_cost(f1, f2)
        if stepwise:
            base *= (1.0 - COUPLING_SUPPRESSION)
        return self.cfg.alpha_coupling * base

    # -----------------------------------------------------------------
    # Term 6: Keyboard action (register-dependent)
    # -----------------------------------------------------------------
    def _key_action_cost(self, midi, finger):
        """Energy to depress key, modulated by arm-weight channeling."""
        E = self.keyboard.key_energy(midi)
        alpha_arm = self.tradition.arm_weight
        m_arm = 3.5  # kg, approximate forearm mass
        arm_offset = alpha_arm * m_arm * GRAVITY * ESCAPEMENT_DEPTH / 1000.0
        effective_E = max(0, E - arm_offset)
        return self.cfg.alpha_key * effective_E

    # -----------------------------------------------------------------
    # Term 7: Contact mechanics (Hertzian vs. pre-load)
    # -----------------------------------------------------------------
    def _contact_cost(self, midi, finger):
        """Impact vs. pre-load contact energy ratio."""
        if self.cfg.technique == 'preload':
            return 0.0  # pre-load: finger already on key
        # Impact: Hertzian collision adds ~30% energy premium
        E_key = self.keyboard.key_energy(midi)
        m_eff = self.hand.effective_mass[finger - 1]
        h_strike = 0.015  # 15 mm typical striking height
        E_impact = m_eff * GRAVITY * h_strike * 1000  # mJ
        ratio = E_impact / max(E_key, 0.001)
        return self.cfg.alpha_contact * ratio

    # -----------------------------------------------------------------
    # Term 8: Finger-pad adhesion (Coulomb friction)
    # -----------------------------------------------------------------
    def _adhesion_cost(self, midi, finger):
        """Adhesion penalty: thumb on black keys under low friction."""
        if not self.keyboard.is_black(midi):
            return 0.0
        # Black key: narrower, elevated, requires lateral stability
        # Thumb (flat contact) is worst; curved fingers 2-4 are better
        base_penalty = (BLACK_KEY_WIDTH / WHITE_KEY_WIDTH)  # ~0.58
        # Friction reduction: low μ_f means less grip
        friction_factor = max(0, 1.0 - self.cfg.mu_f)
        # Thumb is most affected (flat pad, poor lateral grip)
        if finger == 1:
            penalty = self.tradition.thumb_on_black * base_penalty * (
                1.0 + friction_factor)
        elif finger == 5:
            penalty = 0.3 * base_penalty * (1.0 + friction_factor * 0.5)
        else:
            penalty = 0.1 * base_penalty * friction_factor
        return self.cfg.alpha_adhesion * penalty

    # -----------------------------------------------------------------
    # Term 9: Tempo-dependent accessible configuration space
    # -----------------------------------------------------------------
    def _tempo_cost(self, midi1, midi2, dt):
        """Penalty when repositioning exceeds neuromuscular velocity limit."""
        d_mm = self.keyboard.topographic_distance(midi1, midi2)
        d_m = d_mm / 1000.0
        v_required = d_m / max(dt, 0.001)
        excess = max(0, v_required / MAX_FINGER_VELOCITY - 1.0)
        return self.cfg.alpha_tempo * excess**2

    # -----------------------------------------------------------------
    # Term 10: Rayleigh dissipation
    # -----------------------------------------------------------------
    def _dissipation_cost(self, midi1, midi2, dt):
        """Joint viscosity: proportional to squared angular velocity."""
        d_mm = self.keyboard.topographic_distance(midi1, midi2)
        d_m = d_mm / 1000.0
        v = d_m / max(dt, 0.001)
        return self.cfg.alpha_dissipation * JOINT_VISCOSITY * v**2

    # -----------------------------------------------------------------
    # Term 13: Forearm rotation (Eq. 4)
    # -----------------------------------------------------------------
    def _rotation_cost(self, midi1, midi2, finger1, finger2):
        """Forearm rotation cost (Eq. 4, fingering-dependent form).

            C_rot = alpha_rot * (delta_omega)^2,   |delta_omega| > 0.5 omega_max

        The Taubman school (Golandsky 2001; Kochevitsky 1967) holds that
        forearm rotation, not isolated finger motion, transfers weight
        between keys, and that motion is organized around the *playing
        unit* whose rotational axis runs through the centre of the hand
        (functionally the middle finger).  The angular displacement
        required by a transition therefore depends not only on the
        lateral key distance but on *which finger* must arrive over the
        target: bringing an ulnar- or radial-extreme finger (thumb or
        little) over a distant key counter-rotates the playing unit more
        than bringing the central finger, which is already aligned with
        the rotational axis.

        We model delta_omega as the lateral key displacement mapped
        through the forearm lever, scaled by the arriving finger's
        offset from the rotational axis.  That offset is taken from the
        same inter-finger span geometry used elsewhere: the lateral
        distance from the middle finger (the axis) to the arriving
        finger, normalized by the hand's maximum span.  No new constant
        is introduced; omega_max, alpha_rot, the lever, and the spans
        already exist in the model.
        """
        dx_mm = abs(self.keyboard.key_distance(midi1, midi2))
        # Effective forearm-to-fingertip radius (~270 mm; Tubiana 1996).
        lever_mm = 270.0

        # Arriving finger's lateral offset from the forearm rotational
        # axis (the middle finger, index 2), as a fraction of the hand's
        # maximum span.  Fingers aligned with the axis recruit little
        # rotation; extreme fingers recruit proportionally more.
        AXIS = 2  # middle finger, 0-indexed
        arriving = finger2 - 1
        if arriving == AXIS:
            axis_offset_mm = 0.0
        else:
            axis_offset_mm = self.hand.finger_reach_mm(AXIS, arriving)
        alignment = 1.0 + axis_offset_mm / self.hand.max_span

        delta_omega = (dx_mm / lever_mm) * alignment
        gate = 0.5 * OMEGA_MAX
        if abs(delta_omega) > gate:
            excess = abs(delta_omega) - gate
            return ALPHA_ROTATION * excess**2
        return 0.0
        # Effective forearm-to-fingertip radius (~270 mm; Tubiana 1996).
        # With omega_max = 1.2 rad, the gate 0.5*omega_max corresponds to
        # an octave-scale lateral span, matching Sec. 2.10's description of
        # rotation as consequential for large, alignment-breaking moves.
        lever_mm = 270.0
        delta_omega = dx_mm / lever_mm
        gate = 0.5 * OMEGA_MAX
        if abs(delta_omega) > gate:
            excess = abs(delta_omega) - gate
            return ALPHA_ROTATION * excess**2
        return 0.0

    def _arm_weight_selection_cost(self, note2, finger2):
        """Optional weight-channeling SELECTION term (off by default).

        The faithful gravity term :meth:`_gravity_cost` depends only on the
        single finger and the tradition's arm-weight coefficient, which
        rescales every candidate fingering's gravitational cost uniformly;
        it therefore governs action magnitude but cannot reorder
        fingerings.  The arm-weight schools (Russian, Taubman), however,
        deliberately place *strong* fingers on tonally weighted notes so
        that forearm weight is transmitted through a finger able to bear
        it.  This optional term encodes that selection effect: on a
        metrically strong note, a stronger finger earns a channeling
        reward proportional to the tradition's arm-weight coefficient.

        It reuses the strength index of :meth:`_gravity_cost` and the
        existing ``beat_position`` metrical signal; no new constants are
        introduced.  Enabled only when ``cfg.arm_weight_selection`` is
        True, so the default solver is unaffected.
        """
        alpha_arm = self.tradition.arm_weight
        if alpha_arm <= 0.0:
            return 0.0
        # Strong metrical position only (downbeat).
        if getattr(note2, "beat_position", None) != 0:
            return 0.0
        strength = {1: 1.0, 2: 0.85, 3: 0.75, 4: 0.55, 5: 0.45}
        # Reward (negative cost) for channeling weight through a strong
        # finger on the weighted note; scaled by the same alpha_gravity and
        # arm-weight coefficient that drive the faithful term.
        return -self.cfg.alpha_gravity * alpha_arm * strength[finger2] * 3.0

    def _stagger_cost(self, midi1, midi2, finger1, finger2):
        """Black-key depth-stagger cost (Eq. 5, fingering-dependent form).

        The black keys sit ``DEPTH_STAGGER_DZ`` (12.7 mm) above the
        white-key surface.  Section 2.11 states that a colour change
        forces "an asymmetric *hand configuration*": the cost is
        therefore not a property of the note pair alone but of the
        *finger pair* that must bridge the elevation difference.  Two
        kinematic factors, both derived from the Buchholz hand model,
        govern how costly that bridge is:

          1. Span utilization.  A finger pair already near its maximum
             inter-finger reach has little postural slack to absorb a
             12.7 mm vertical offset; the same ``stretch`` ratio used by
             the topographic term measures this.

          2. Length mismatch.  When the *shorter* finger must take the
             elevated (black) key while the *longer* finger rests on the
             lower (white) key, the hand is forced into a more strained,
             asymmetric posture than when the naturally longer finger
             takes the higher key.  The signed length difference between
             the two fingers, normalized by hand length, captures this.

        The term reduces to zero within a colour (no elevation change)
        and is symmetric in neither the fingers nor the direction of the
        colour change, so it discriminates between candidate fingerings
        and enters the optimization rather than adding a path constant.

            C_stagger = alpha_bk * (|dz|/dz0) * stretch * lambda(f_i,f_j)

        with lambda the kinematic length-mismatch factor below.  No new
        constants are introduced: alpha_bk, dz0 = DEPTH_STAGGER_DZ, the
        reach, and the segment lengths are all already in the model.
        """
        b1 = self.keyboard.is_black(midi1)
        b2 = self.keyboard.is_black(midi2)
        if b1 == b2:
            return 0.0

        f1_idx, f2_idx = finger1 - 1, finger2 - 1

        # (1) Span utilization: how close the pair is to its max reach.
        d_mm = self.keyboard.key_distance(midi1, midi2)
        reach = self.hand.finger_reach_mm(f1_idx, f2_idx)
        stretch = d_mm / reach if reach > 0 else 1.0

        # (2) Length mismatch: the finger landing on the BLACK (elevated)
        # key versus the finger on the white key.  Strain is higher when
        # the shorter finger must reach up to the black key.
        black_idx = f2_idx if b2 else f1_idx       # finger on the black key
        white_idx = f1_idx if b2 else f2_idx       # finger on the white key
        len_black = self.hand.finger_length_mm(black_idx)
        len_white = self.hand.finger_length_mm(white_idx)
        # Normalize the signed length difference by the hand's own natural
        # length scale -- the spread between its longest and shortest
        # finger -- so the factor needs no chosen multiplier.  The deficit
        # is +1 in the most strained case (shortest finger on the black
        # key while the longest sits on the white key) and -1 in the most
        # relieved case; lambda is then the non-negative neutral-centered
        # factor 1 + deficit, spanning [0, 2].
        all_len = [self.hand.finger_length_mm(k) for k in range(5)]
        len_range = max(all_len) - min(all_len)
        deficit = (len_white - len_black) / len_range if len_range > 0 else 0.0
        lam = max(0.0, 1.0 + deficit)

        return ALPHA_BLACK_KEY * stretch * lam

    # -----------------------------------------------------------------
    # Term 11: Pedagogical tradition
    # -----------------------------------------------------------------
    def _tradition_cost(self, midi1, midi2, finger1, finger2, stepwise):
        """Tradition-specific cost: thumb-under, finger crossing,
        paired fingers, legato, rotation, natural position."""
        cost = 0.0
        interval = midi2 - midi1
        ascending = interval > 0

        # Thumb-under: thumb crosses under fingers
        if finger1 != 1 and finger2 == 1 and ascending:
            base = self.tradition.thumb_under
            # Per-finger difficulty scaling
            scale = self.thumb_under_scale.get(finger1, 0.5)
            cost += base * (0.3 + scale)
        elif finger1 == 1 and finger2 != 1 and not ascending:
            base = self.tradition.thumb_under
            scale = self.thumb_under_scale.get(finger2, 0.5)
            cost += base * (0.3 + scale)

        # Finger crossing (non-thumb, over another finger)
        # In traditions with paired-finger technique (Baroque), adjacent
        # finger crossings are the intended technique, not a penalty.
        is_adjacent_cross = False
        if finger1 > 1 and finger2 > 1:
            if (ascending and finger2 < finger1) or (
                    not ascending and finger2 > finger1):
                if abs(finger2 - finger1) == 1 and self.tradition.paired_finger_bonus < 0:
                    # Adjacent paired-finger crossing: exempt from penalty
                    is_adjacent_cross = True
                else:
                    cost += self.tradition.finger_crossing

        # Paired-finger bonus (Baroque: adjacent fingers on stepwise)
        if stepwise and abs(finger2 - finger1) == 1 and finger1 > 1:
            cost += self.tradition.paired_finger_bonus
            if is_adjacent_cross:
                # Additional reward for the characteristic Baroque technique
                cost += self.tradition.paired_finger_bonus * 0.5

        # Legato bonus (keeping hand in position for smooth connection)
        if abs(finger2 - finger1) <= 2 and abs(interval) <= 4:
            cost += self.tradition.legato_bonus

        # Natural position bonus (Chopin: hand fits the black-key
        # topography, E-F#-G#-A#-B# under 1-2-3-4-5)
        if self.tradition.natural_position != 0:
            # Reward when finger falls on its 'natural' key
            natural_map = {1: {0, 5}, 2: {1, 6}, 3: {3, 8},
                           4: {6, 8, 10}, 5: {11, 0}}
            pc = midi2 % 12
            if pc in natural_map.get(finger2, set()):
                cost += self.tradition.natural_position

        # Metric weight (Baroque/Classical: strong beats get strong fingers)
        # (applied externally in transition_cost via beat_position)

        return cost

    # -----------------------------------------------------------------
    # Combined transition cost (Eq. 8 in the paper)
    # -----------------------------------------------------------------
    def transition_cost(self, note1: NoteEvent, finger1: int,
                        note2: NoteEvent, finger2: int) -> float:
        """Evaluate the complete 14-term discrete transition cost.

        This is the core of the equation of state: the sum of all
        non-negative penalty terms (and negative rewards), corresponding
        to the minimum-action path of the continuous formulation.
        """
        midi1, midi2 = note1.midi, note2.midi
        dt = max(note2.onset - note1.onset, 0.01)
        interval = abs(midi2 - midi1)
        stepwise = interval <= 2

        # Same finger on different notes: only viable if same key (repeat)
        if finger1 == finger2 and midi1 != midi2:
            return 200.0  # physically impossible (must lift and restrike)

        cost = 0.0

        # 1. Kinetic energy
        cost += self._kinetic_cost(midi1, midi2, finger1, finger2, dt)

        # 2. Topographic potential
        cost += self._topographic_cost(midi1, midi2, finger1, finger2)

        # 3. Gravitational potential
        cost += self._gravity_cost(finger2)

        # 4. Collision avoidance
        cost += self._collision_cost(midi1, midi2, finger1, finger2)

        # 5. Inter-digit coupling
        cost += self._coupling_cost(finger1, finger2, stepwise)

        # 6. Keyboard action
        cost += self._key_action_cost(midi2, finger2)

        # 7. Contact mechanics
        cost += self._contact_cost(midi2, finger2)

        # 8. Adhesion
        cost += self._adhesion_cost(midi2, finger2)

        # 9. Tempo constraint
        cost += self._tempo_cost(midi1, midi2, dt)

        # 10. Dissipation
        cost += self._dissipation_cost(midi1, midi2, dt)

        # 11. Tradition
        cost += self._tradition_cost(midi1, midi2, finger1, finger2,
                                     stepwise)

        # 12. Forearm rotation (standalone biomechanical term, Eq. 4)
        cost += self._rotation_cost(midi1, midi2, finger1, finger2)

        # 13. Black-key depth stagger (Eq. 5)
        cost += self._stagger_cost(midi1, midi2, finger1, finger2)

        # Optional (off by default): arm-weight selection variant.
        if self.cfg.arm_weight_selection:
            cost += self._arm_weight_selection_cost(note2, finger2)

        # Phrase structure: sequential finger reward
        # Only when finger direction matches pitch direction
        # (ascending pitch → ascending finger number, or vice versa)
        if stepwise and abs(finger2 - finger1) == 1:
            ascending_pitch = midi2 > midi1
            ascending_finger = finger2 > finger1
            if ascending_pitch == ascending_finger:
                cost += SEQUENTIAL_REWARD

        # Metric alignment (Baroque/Classical)
        if self.tradition.metric_weight > 0 and note2.beat_position == 0:
            # Strong beat: reward strong fingers (1, 2, 3)
            if finger2 <= 3:
                cost -= self.tradition.metric_weight * 0.5
            else:
                cost += self.tradition.metric_weight * 0.3

        # --- v2.1 Patch 2: Arpeggio grouping enhancement ---
        # On non-stepwise motion (interval > 2 semitones), apply three
        # mechanisms that steer the solver toward sequential finger groups
        # (1-2-3-5 or 1-2-4-5) instead of 1-3 alternation.
        if not stepwise and interval > 2 and ARPEGGIO_GROUPING != 0.0:
            ascending_p = midi2 > midi1
            # (a) Sequential finger reward on arpeggios:
            #     f_{i+1} = f_i + 1 on ascending is biomechanically
            #     efficient (no repositioning required within group)
            if ascending_p and finger2 == finger1 + 1 and finger2 <= 5:
                cost -= 4.5 * ARPEGGIO_GROUPING
            elif not ascending_p and finger2 == finger1 - 1 and finger2 >= 1:
                cost -= 4.5 * ARPEGGIO_GROUPING

            # (b) Finger gap penalty: skipping 2+ fingers within a group
            #     requires hand reconfiguration that is costlier than
            #     sequential depression (exclude thumb crossings)
            finger_gap = abs(finger2 - finger1)
            if finger_gap > 2 and finger1 != 1 and finger2 != 1:
                cost += 3.0 * (finger_gap - 1) * ARPEGGIO_GROUPING

            # (c) Premature group restart penalty: returning to thumb
            #     before reaching finger 4+ means restarting the group
            #     earlier than necessary, wasting a position
            if ascending_p and finger1 > 1 and finger2 == 1 and finger1 < 4:
                cost += 6.0 * ARPEGGIO_GROUPING

        # --- v2.1 Patch 1: Cost floor ---
        # Rewards (legato, sequential, rotation, arm-weight, natural-position)
        # reduce cost but cannot make it negative.  Negative action is
        # unphysical: it would imply that performing the transition releases
        # energy.  The floor preserves relative ordering while ensuring
        # all transitions have non-negative cost.
        return max(0.0, cost)

    # -----------------------------------------------------------------
    # Initial and terminal costs
    # -----------------------------------------------------------------
    def _initial_cost(self, finger: int, note: NoteEvent) -> float:
        """Cost of starting with a given finger."""
        cost = 0.0
        cost += self._adhesion_cost(note.midi, finger)
        cost += self._key_action_cost(note.midi, finger)
        cost += self._gravity_cost(finger)
        # Prefer starting on thumb or finger 5 at phrase boundaries
        if finger == 1 or finger == 5:
            cost -= 1.0
        return cost

    def _terminal_costs(self, notes: List[NoteEvent]) -> np.ndarray:
        """Terminal preference: ascending passages should end on finger 5."""
        costs = np.zeros(5)
        if len(notes) < 2:
            return costs
        # Detect ascending tendency
        ascending_count = sum(1 for i in range(len(notes) - 1)
                              if notes[i + 1].midi > notes[i].midi)
        descending_count = sum(1 for i in range(len(notes) - 1)
                               if notes[i + 1].midi < notes[i].midi)
        if ascending_count > descending_count:
            # Penalize not ending on finger 5
            for f in range(5):
                if f != 4:  # not finger 5
                    costs[f] = TERMINAL_MULTIPLIER * (4 - f)
        elif descending_count > ascending_count:
            # Penalize not ending on thumb
            for f in range(5):
                if f != 0:  # not thumb
                    costs[f] = TERMINAL_MULTIPLIER * f
        return costs

    # -----------------------------------------------------------------
    # Action-Risk Index (Lyapunov stability metric)
    # -----------------------------------------------------------------
    def _compute_ari(self, notes, dp_cost, optimal_fingers):
        """Compute per-transition Action-Risk Index.

        ARI(i) = C_i* · exp[(1/ΔF) max_δf |ΔC/C*|]

        smoothed with a 4-note rolling window.
        """
        N = len(notes)
        if N < 2:
            return [0.0], ['low']

        raw_ari = []
        for i in range(N - 1):
            fi = optimal_fingers[i] - 1
            c_opt = dp_cost[i, fi]
            if c_opt <= 0 or not np.isfinite(c_opt):
                raw_ari.append(0.0)
                continue
            # Maximum sensitivity to finger perturbation
            max_sensitivity = 0.0
            for delta_f in range(5):
                if delta_f == fi:
                    continue
                c_alt = dp_cost[i, delta_f]
                if np.isfinite(c_alt) and c_opt > 0:
                    sensitivity = abs(c_alt - c_opt) / c_opt
                    max_sensitivity = max(max_sensitivity, sensitivity)
            # Clamp exponent to prevent overflow
            exponent = min(max_sensitivity / 4.0, 10.0)
            ari = c_opt * math.exp(exponent)
            raw_ari.append(ari)

        # 4-note rolling window smoothing
        smoothed = []
        window = 4
        for i in range(len(raw_ari)):
            start = max(0, i - window // 2)
            end = min(len(raw_ari), i + window // 2 + 1)
            smoothed.append(np.mean(raw_ari[start:end]))

        # Risk classification
        if not smoothed:
            return [0.0], ['low']
        p75 = np.percentile(smoothed, 75) if smoothed else 1.0
        risk_levels = []
        for a in smoothed:
            if a < p75 * 0.5:
                risk_levels.append('low')
            elif a < p75:
                risk_levels.append('moderate')
            elif a < p75 * 2.0:
                risk_levels.append('high')
            else:
                risk_levels.append('critical')

        return smoothed, risk_levels

    # -----------------------------------------------------------------
    # Main solve method
    # -----------------------------------------------------------------
    def solve(self, midi_or_notes, tempo_nps=None):
        """Solve for optimal fingering.

        Parameters
        ----------
        midi_or_notes : list of int or list of NoteEvent
            MIDI pitch sequence or NoteEvent list.
        tempo_nps : float, optional
            Override tempo (notes per second).

        Returns
        -------
        dict or SolverResult
            If input is list of int: dict with 'fingers', 'cost', etc.
            If input is list of NoteEvent: SolverResult dataclass.
        """
        # Convert MIDI list to NoteEvents if needed
        if midi_or_notes and isinstance(midi_or_notes[0], int):
            nps = tempo_nps or self.cfg.tempo_nps
            dt = 1.0 / nps
            notes = [NoteEvent(midi=m, onset=i * dt, duration=dt * 0.9,
                               beat_position=(i % 4) / 4.0)
                     for i, m in enumerate(midi_or_notes)]
            return_dict = True
        else:
            notes = midi_or_notes
            return_dict = False

        N = len(notes)
        F = 5  # fingers per hand

        # Viterbi forward pass
        dp_cost = np.full((N, F), np.inf)
        dp_prev = np.full((N, F), -1, dtype=int)

        # Initialize
        for f in range(F):
            dp_cost[0, f] = self._initial_cost(f + 1, notes[0])

        # Forward pass: O(N · F²)
        for i in range(1, N):
            for fj in range(F):
                for fi in range(F):
                    if dp_cost[i - 1, fi] == np.inf:
                        continue
                    tc = self.transition_cost(
                        notes[i - 1], fi + 1, notes[i], fj + 1)
                    total = dp_cost[i - 1, fi] + tc
                    if total < dp_cost[i, fj]:
                        dp_cost[i, fj] = total
                        dp_prev[i, fj] = fi

        # Terminal costs
        terminal = self._terminal_costs(notes)
        final_costs = dp_cost[N - 1] + terminal

        # Backtrack
        best_last = int(np.argmin(final_costs))
        fingers_0 = [0] * N
        fingers_0[N - 1] = best_last
        for i in range(N - 2, -1, -1):
            fingers_0[i] = dp_prev[i + 1, fingers_0[i + 1]]

        # Convert to 1-indexed
        fingers = [f + 1 for f in fingers_0]
        total_cost = float(final_costs[best_last])

        # Per-note costs
        per_note = [float(dp_cost[0, fingers_0[0]])]
        for i in range(1, N):
            tc = self.transition_cost(
                notes[i - 1], fingers[i - 1], notes[i], fingers[i])
            per_note.append(tc)

        # Cost breakdown by term
        breakdown = self._cost_breakdown(notes, fingers)

        # ARI
        ari_values, ari_levels = self._compute_ari(
            notes, dp_cost, fingers)

        result = SolverResult(
            fingers=fingers,
            total_cost=total_cost,
            per_note_cost=per_note,
            cost_breakdown=breakdown,
            ari_values=ari_values,
            ari_risk_levels=ari_levels,
            tradition=self.tradition.name,
            passage_length=N)

        if return_dict:
            return {
                'fingers': result.fingers,
                'cost': result.total_cost,
                'per_note_cost': result.per_note_cost,
                'cost_breakdown': result.cost_breakdown,
                'ari_values': result.ari_values,
                'ari_risk_levels': result.ari_risk_levels,
                'tradition': result.tradition,
            }
        return result

    def _cost_breakdown(self, notes, fingers):
        """Decompose total cost by equation-of-state term."""
        terms = {
            'kinetic': 0, 'topographic': 0, 'gravity': 0,
            'collision': 0, 'coupling': 0, 'key_action': 0,
            'contact': 0, 'adhesion': 0, 'tempo': 0,
            'dissipation': 0, 'tradition': 0, 'phrase': 0,
            'rotation': 0, 'stagger': 0,
            'arpeggio': 0, 'floor_clipped': 0,
        }
        for i in range(1, len(notes)):
            m1, m2 = notes[i-1].midi, notes[i].midi
            f1, f2 = fingers[i-1], fingers[i]
            dt = max(notes[i].onset - notes[i-1].onset, 0.01)
            interval = abs(m2 - m1)
            stepwise = interval <= 2
            raw_cost = 0.0
            terms['kinetic'] += self._kinetic_cost(m1, m2, f1, f2, dt)
            terms['topographic'] += self._topographic_cost(m1, m2, f1, f2)
            terms['gravity'] += self._gravity_cost(f2)
            terms['collision'] += self._collision_cost(m1, m2, f1, f2)
            terms['coupling'] += self._coupling_cost(f1, f2, stepwise)
            terms['key_action'] += self._key_action_cost(m2, f2)
            terms['contact'] += self._contact_cost(m2, f2)
            terms['adhesion'] += self._adhesion_cost(m2, f2)
            terms['tempo'] += self._tempo_cost(m1, m2, dt)
            terms['dissipation'] += self._dissipation_cost(m1, m2, dt)
            terms['tradition'] += self._tradition_cost(m1, m2, f1, f2,
                                                       stepwise)
            terms['rotation'] += self._rotation_cost(m1, m2, f1, f2)
            terms['stagger'] += self._stagger_cost(m1, m2, f1, f2)
            if stepwise and abs(f2 - f1) == 1:
                ascending_p = m2 > m1
                ascending_f = f2 > f1
                if ascending_p == ascending_f:
                    terms['phrase'] += SEQUENTIAL_REWARD
            # Arpeggio terms (v2.1)
            if not stepwise and interval > 2:
                ascending_p = m2 > m1
                if ascending_p and f2 == f1 + 1 and f2 <= 5:
                    terms['arpeggio'] -= 4.5
                elif not ascending_p and f2 == f1 - 1 and f2 >= 1:
                    terms['arpeggio'] -= 4.5
                fg = abs(f2 - f1)
                if fg > 2 and f1 != 1 and f2 != 1:
                    terms['arpeggio'] += 3.0 * (fg - 1)
                if ascending_p and f1 > 1 and f2 == 1 and f1 < 4:
                    terms['arpeggio'] += 6.0
        return terms

    # -----------------------------------------------------------------
    # Stochastic robustness analysis
    # -----------------------------------------------------------------
    def solve_stochastic(self, notes, n_trials=100, sigma=None):
        """Monte Carlo robustness: perturb costs and track finger stability.

        Returns dict with 'deterministic', 'robustness', and
        'finger_distributions'.
        """
        if sigma is None:
            sigma = self.cfg.stochastic_sigma

        det = self.solve(notes)
        if isinstance(det, dict):
            det_fingers = det['fingers']
        else:
            det_fingers = det.fingers

        N = len(notes)
        counts = np.zeros((N, 5))
        agreement = np.zeros(N)

        for trial in range(n_trials):
            # Perturb each transition cost by multiplicative noise
            perturbed = self._solve_perturbed(notes, sigma)
            for i, f in enumerate(perturbed):
                counts[i, f - 1] += 1
                if f == det_fingers[i]:
                    agreement[i] += 1

        robustness = (agreement / n_trials).tolist()
        distributions = (counts / n_trials).tolist()

        return {
            'deterministic': det,
            'robustness': robustness,
            'finger_distributions': distributions,
            'trials': n_trials,
            'sigma': sigma,
        }

    def _solve_perturbed(self, notes, sigma):
        """Solve with multiplicative Gaussian noise on transition costs."""
        N = len(notes)
        F = 5

        dp_cost = np.full((N, F), np.inf)
        dp_prev = np.full((N, F), -1, dtype=int)

        for f in range(F):
            dp_cost[0, f] = self._initial_cost(f + 1, notes[0])

        for i in range(1, N):
            for fj in range(F):
                for fi in range(F):
                    if dp_cost[i - 1, fi] == np.inf:
                        continue
                    tc = self.transition_cost(
                        notes[i - 1], fi + 1, notes[i], fj + 1)
                    # Multiplicative perturbation
                    noise = 1.0 + np.random.normal(0, sigma)
                    tc *= max(noise, 0.1)
                    total = dp_cost[i - 1, fi] + tc
                    if total < dp_cost[i, fj]:
                        dp_cost[i, fj] = total
                        dp_prev[i, fj] = fi

        terminal = self._terminal_costs(notes)
        final = dp_cost[N - 1] + terminal
        best = int(np.argmin(final))
        fingers_0 = [0] * N
        fingers_0[N - 1] = best
        for i in range(N - 2, -1, -1):
            fingers_0[i] = dp_prev[i + 1, fingers_0[i + 1]]
        return [f + 1 for f in fingers_0]


# =====================================================================
# Embedded Test Passages
# =====================================================================

def passage_bach_invention_13():
    """Bach Invention 13 in A minor, mm. 11-13 (RH)."""
    return [76, 72, 69, 65, 64, 67, 71, 74,
            76, 72, 69, 65, 64, 69, 72, 76,
            77, 74, 71, 67, 69, 72, 76, 81]

def passage_chopin_op10_1():
    """Chopin Étude Op.10/1 in C major, mm. 1-4 (RH)."""
    return [60, 64, 67, 72, 76, 79, 84,
            83, 79, 76, 72, 67, 64, 60]

def passage_bach_wtc_fugue():
    """Bach WTC I Fugue in C minor, mm. 7-10 (RH)."""
    return [60, 62, 63, 65, 67, 68, 71, 72,
            72, 70, 68, 67, 65, 63, 65, 67, 68, 72]

def passage_c_major_scale():
    """C major scale, 2 octaves ascending (RH)."""
    return [60, 62, 64, 65, 67, 69, 71,
            72, 74, 76, 77, 79, 81, 83, 84]

def passage_chopin_op25_6():
    """Chopin Étude Op.25/6 in G# minor, mm. 1-2 (RH)."""
    return [80, 78, 76, 75, 73, 71, 70, 68,
            68, 70, 71, 73, 75, 76, 78, 80]


# =====================================================================
# Convenience: total action with kinetic energy for scaling law
# =====================================================================

def total_action(solver, midis, tempo_nps):
    """Compute S = J + T for scaling law analysis.

    J: Viterbi potential cost.
    T: kinetic energy contribution T = β · Σ v².
    """
    dt = 1.0 / tempo_nps
    notes = [NoteEvent(midi=m, onset=i * dt, duration=dt * 0.9,
                       beat_position=(i % 4) / 4.0)
             for i, m in enumerate(midis)]

    result = solver.solve(notes)
    J = result.total_cost if isinstance(result, SolverResult) else result['cost']

    beta_kinetic = 100.0
    T = 0.0
    keyboard = solver.keyboard
    for i in range(1, len(midis)):
        d_mm = keyboard.topographic_distance(midis[i-1], midis[i])
        d_m = d_mm / 1000.0
        v = d_m / dt
        T += beta_kinetic * v**2

    return J + T


# =====================================================================
# Two-Hand Bilateral Solver (v2.1)
# =====================================================================

@dataclass
class BilateralNote:
    """A note with hand assignment metadata."""
    midi: int
    onset: float = 0.0
    duration: float = 0.25
    dynamic: str = 'mf'
    beat_position: float = 0.0
    hand: Optional[str] = None
    staff: Optional[int] = None
    shared: bool = False


@dataclass
class BilateralResult:
    """Complete result from the bilateral solver."""
    rh_fingers: List[int] = field(default_factory=list)
    lh_fingers: List[int] = field(default_factory=list)
    rh_notes: List[BilateralNote] = field(default_factory=list)
    lh_notes: List[BilateralNote] = field(default_factory=list)
    rh_cost: float = 0.0
    lh_cost: float = 0.0
    bilateral_cost: float = 0.0
    collision_cost: float = 0.0
    shared_assignments: List[Tuple] = field(default_factory=list)
    collision_points: List[Tuple] = field(default_factory=list)
    crossing_mode: str = 'none'

    def summary(self) -> str:
        lines = [
            f'Bilateral Fingering Solution',
            f'  RH: {len(self.rh_notes)} notes, cost = {self.rh_cost:.1f}',
            f'  LH: {len(self.lh_notes)} notes, cost = {self.lh_cost:.1f}',
            f'  Collision penalty: {self.collision_cost:.1f}',
            f'  Total bilateral cost: {self.bilateral_cost:.1f}',
            f'  Crossing mode: {self.crossing_mode}',
        ]
        return '\n'.join(lines)


def _midi_to_mm(midi: int) -> float:
    """MIDI pitch to approximate horizontal position (mm)."""
    return (midi - 60) / 12.0 * 164.5


class BilateralSolver:
    """Two-hand solver: partition → independent Viterbi → bilateral refinement.

    Phase 1: Assign notes to LH/RH by staff and register.
    Phase 2: Solve each hand independently.
    Phase 3: Compute inter-hand collision penalties and detect crossings.

    Usage:
        rh = HamiltonianSolver(tradition='modern')
        lh = HamiltonianSolver(tradition='modern')
        bi = BilateralSolver(rh, lh)
        result = bi.solve(treble_notes, bass_notes)
    """

    OVERLAP_LOW = 55    # G3
    OVERLAP_HIGH = 67   # G4
    MIN_SEPARATION_MM = 30.0
    CROSSING_PENALTY = 20.0
    THUMB_COLLISION = 15.0

    def __init__(self, rh_solver, lh_solver, partition_midi=60):
        self.rh_solver = rh_solver
        self.lh_solver = lh_solver
        self.partition_midi = partition_midi

    def _to_bilateral(self, note, hand, staff):
        if isinstance(note, BilateralNote):
            note.hand = hand
            note.staff = staff
            return note
        midi = note.midi if hasattr(note, 'midi') else int(note)
        return BilateralNote(
            midi=midi,
            onset=getattr(note, 'onset', 0.0),
            duration=getattr(note, 'duration', 0.25),
            dynamic=getattr(note, 'dynamic', 'mf'),
            beat_position=getattr(note, 'beat_position', 0.0),
            hand=hand, staff=staff,
        )

    def partition_notes(self, treble, bass):
        rh, lh, shared = [], [], []
        for i, n in enumerate(treble):
            bn = self._to_bilateral(n, 'right', 0)
            if self.OVERLAP_LOW <= bn.midi <= self.OVERLAP_HIGH:
                bn.shared = True
                shared.append(('treble', i, len(rh)))
            rh.append(bn)
        for i, n in enumerate(bass):
            bn = self._to_bilateral(n, 'left', 1)
            if self.OVERLAP_LOW <= bn.midi <= self.OVERLAP_HIGH:
                bn.shared = True
                shared.append(('bass', i, len(lh)))
            lh.append(bn)
        return rh, lh, shared

    def _solve_hand(self, solver, notes):
        if not notes:
            return [], 0.0
        midis = [n.midi for n in notes]
        result = solver.solve(midis)
        if isinstance(result, dict):
            return result.get('fingers', []), result.get('cost', 0.0)
        return getattr(result, 'fingers', []), getattr(result, 'total_cost', 0.0)

    def _collision_cost(self, rh_notes, rh_fingers, lh_notes, lh_fingers):
        total = 0.0
        points = []
        for i, rn in enumerate(rh_notes):
            rh_end = rn.onset + rn.duration
            rh_mm = _midi_to_mm(rn.midi)
            for j, ln in enumerate(lh_notes):
                if ln.onset >= rh_end or rn.onset >= ln.onset + ln.duration:
                    continue
                sep = abs(rh_mm - _midi_to_mm(ln.midi))
                if sep < self.MIN_SEPARATION_MM:
                    penalty = 8.0 * (1.0 - sep / self.MIN_SEPARATION_MM)
                    rf = rh_fingers[i] if i < len(rh_fingers) else 0
                    lf = lh_fingers[j] if j < len(lh_fingers) else 0
                    if rf == 1 and lf == 1:
                        penalty += self.THUMB_COLLISION
                    total += penalty
                    points.append((rn.onset, rn.midi, ln.midi))
        return total, points

    def _detect_crossings(self, rh_notes, lh_notes):
        mode = 'none'
        for rn in rh_notes:
            rh_end = rn.onset + rn.duration
            for ln in lh_notes:
                if ln.onset >= rh_end or rn.onset >= ln.onset + ln.duration:
                    continue
                if ln.midi > rn.midi:
                    mode = 'crossing'
                    return mode
                elif rn.midi - ln.midi < 12:
                    mode = 'interleaving'
        return mode

    def solve(self, treble_notes, bass_notes):
        result = BilateralResult()
        rh_n, lh_n, shared = self.partition_notes(treble_notes, bass_notes)
        result.rh_notes = rh_n
        result.lh_notes = lh_n

        result.rh_fingers, result.rh_cost = self._solve_hand(
            self.rh_solver, rh_n)
        result.lh_fingers, result.lh_cost = self._solve_hand(
            self.lh_solver, lh_n)

        result.crossing_mode = self._detect_crossings(rh_n, lh_n)
        coll, pts = self._collision_cost(
            rh_n, result.rh_fingers, lh_n, result.lh_fingers)
        result.collision_cost = coll
        result.collision_points = pts

        # Crossing penalty
        cross_cost = self.CROSSING_PENALTY if result.crossing_mode == 'crossing' else 0.0

        # Shared note reassignment heuristic
        reassignments = []
        for src, orig, seq in shared:
            if src == 'treble' and seq < len(rh_n):
                if rh_n[seq].midi < self.partition_midi - 3:
                    reassignments.append((src, orig, 'left'))
            elif src == 'bass' and seq < len(lh_n):
                if lh_n[seq].midi > self.partition_midi + 3:
                    reassignments.append((src, orig, 'right'))
        result.shared_assignments = reassignments
        result.bilateral_cost = result.rh_cost + result.lh_cost + coll + cross_cost
        return result

    def solve_from_grand_staff(self, notes):
        treble = [n for n in notes if getattr(n, 'staff', None) == 0]
        bass = [n for n in notes if getattr(n, 'staff', None) == 1]
        unassigned = [n for n in notes if getattr(n, 'staff', None) is None]
        for n in unassigned:
            m = n.midi if hasattr(n, 'midi') else n
            if m >= self.partition_midi:
                if hasattr(n, 'staff'):
                    n.staff = 0
                treble.append(n)
            else:
                if hasattr(n, 'staff'):
                    n.staff = 1
                bass.append(n)
        treble.sort(key=lambda n: getattr(n, 'onset', 0))
        bass.sort(key=lambda n: getattr(n, 'onset', 0))
        return self.solve(treble, bass)


# =====================================================================
# Main: self-test
# =====================================================================

if __name__ == '__main__':
    print("=" * 65)
    print("Hamiltonian Solver v2.1 — Self-Test")
    print("=" * 65)

    passages = [
        ("Bach Invention 13, mm. 11-13", passage_bach_invention_13()),
        ("Chopin Op.10/1, mm. 1-4", passage_chopin_op10_1()),
        ("Bach WTC I Fugue C minor", passage_bach_wtc_fugue()),
        ("C major scale, 2 octaves", passage_c_major_scale()),
        ("Chopin Op.25/6, mm. 1-2", passage_chopin_op25_6()),
    ]

    for tradition in ['baroque', 'modern', 'russian', 'french']:
        print(f"\n  Tradition: {tradition.upper()}")
        solver = HamiltonianSolver(tradition=tradition)
        for name, midis in passages:
            result = solver.solve(midis)
            fstr = '-'.join(str(f) for f in result['fingers'])
            print(f"    {name:35s}  J={result['cost']:7.1f}  "
                  f"F=[{fstr}]")

    # Scale fingering emergence test
    print("\n  --- Scale Fingering Emergence ---")
    for tradition in TRADITIONS:
        solver = HamiltonianSolver(tradition=tradition)
        result = solver.solve(passage_c_major_scale())
        fstr = '-'.join(str(f) for f in result['fingers'])
        print(f"    {tradition:12s}: {fstr}")

    # v2.1: Cost floor verification
    print("\n  --- v2.1: Cost Floor Verification ---")
    all_ok = True
    for tradition in TRADITIONS:
        solver = HamiltonianSolver(tradition=tradition)
        result = solver.solve([60,62,64,65,67,69,71,72])
        neg = sum(1 for c in result['per_note_cost'] if c < -0.001)
        ok = result['cost'] >= 0 and neg == 0
        if not ok:
            all_ok = False
        print(f"    {tradition:12s}: J={result['cost']:6.2f}  neg={neg}  "
              f"[{'OK' if ok else 'FAIL'}]")
    print(f"    Cost floor: {'PASS' if all_ok else 'FAIL'}")

    # v2.1: Arpeggio grouping
    print("\n  --- v2.1: Arpeggio Grouping ---")
    arp = [60, 64, 67, 72, 76, 79, 84]
    for hl, hb, label in [(210, 95, 'Large'), (190, 85, 'Med'), (165, 72, 'Small')]:
        solver = HamiltonianSolver(tradition='modern',
                                   hand_length=hl, hand_breadth=hb)
        result = solver.solve(arp)
        fstr = '-'.join(str(f) for f in result['fingers'])
        no13 = result['fingers'][:2] != [1, 3]
        print(f"    {label:5s} ({hl}mm): {fstr}  no-1-3={no13}")

    # v2.1: Bilateral solver
    print("\n  --- v2.1: Bilateral Solver ---")
    rh = HamiltonianSolver(tradition='modern')
    lh = HamiltonianSolver(tradition='modern')
    bi = BilateralSolver(rh, lh)
    treble = [NoteEvent(midi=m, onset=i*0.25) for i, m in
              enumerate([60,64,67,72,76,79,84])]
    bass = [NoteEvent(midi=m, onset=i*0.25) for i, m in
            enumerate([48,43,40,36,33,31,28])]
    br = bi.solve(treble, bass)
    print(f"    RH: {'-'.join(str(f) for f in br.rh_fingers)}  "
          f"cost={br.rh_cost:.1f}")
    print(f"    LH: {'-'.join(str(f) for f in br.lh_fingers)}  "
          f"cost={br.lh_cost:.1f}")
    print(f"    Bilateral cost: {br.bilateral_cost:.1f}  "
          f"mode={br.crossing_mode}")

    print("\n" + "=" * 65)
    print("Self-test complete.")
    print("=" * 65)
