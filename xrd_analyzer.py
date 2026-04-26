# XRD Peak Analyzer
# Author: Aashritha Narala | MS MSE, NC State University
# Description: Automated XRD peak detection, d-spacing calculation using Bragg's Law,
#              and publication-quality plot generation for crystal structure analysis.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# ── Bragg's Law ──────────────────────────────────────────────────────────────
def bragg_d_spacing(two_theta_deg, wavelength=1.5406):
    """
    Calculate d-spacing using Bragg's Law: nλ = 2d·sin(θ)
    Default wavelength: Cu Kα = 1.5406 Å
    """
    theta_rad = np.radians(two_theta_deg / 2)
    return wavelength / (2 * np.sin(theta_rad))

# ── Gaussian fit for peak profiling ─────────────────────────────────────────
def gaussian(x, amp, center, sigma):
    return amp * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))

def fit_peak(two_theta, intensity, peak_idx, window=1.5):
    mask = (two_theta >= two_theta[peak_idx] - window) & \
           (two_theta <= two_theta[peak_idx] + window)
    x, y = two_theta[mask], intensity[mask]
    try:
        popt, _ = curve_fit(gaussian, x, y,
                            p0=[intensity[peak_idx], two_theta[peak_idx], 0.5],
                            maxfev=2000)
        return popt  # amp, center, sigma
    except Exception:
        return None

# ── Generate synthetic XRD data (Iron BCC α-Fe) ─────────────────────────────
def generate_iron_xrd(noise_level=0.03):
    """
    Simulates α-Fe (BCC) XRD pattern with Cu Kα radiation.
    Real peaks at: 44.67°, 65.02°, 82.33°, 98.94° (2θ)
    """
    two_theta = np.linspace(20, 110, 2000)
    intensity = np.zeros_like(two_theta)

    peaks_data = [
        (44.67, 1.00, 0.35),   # (110) — strongest
        (65.02, 0.20, 0.40),   # (200)
        (82.33, 0.35, 0.38),   # (211)
        (98.94, 0.12, 0.42),   # (220)
    ]

    for center, rel_amp, sigma in peaks_data:
        intensity += rel_amp * np.exp(-((two_theta - center) ** 2) / (2 * sigma ** 2))

    # Background + noise
    background = 0.02 + 0.001 * (two_theta - 20)
    noise = np.random.normal(0, noise_level, len(two_theta))
    intensity = intensity + background + np.abs(noise)
    return two_theta, intensity / intensity.max()

# ── Main Analyzer ─────────────────────────────────────────────────────────────
def analyze_xrd(two_theta, intensity, wavelength=1.5406,
                height_threshold=0.25, prominence=0.20):

    # Detect peaks
    peaks_idx, properties = find_peaks(intensity,
                                       height=height_threshold,
                                       prominence=prominence,
                                       distance=30)

    results = []
    for idx in peaks_idx:
        two_theta_val = two_theta[idx]
        d = bragg_d_spacing(two_theta_val, wavelength)
        rel_intensity = intensity[idx] / intensity[peaks_idx[0]] * 100
        fit = fit_peak(two_theta, intensity, idx)
        fwhm = fit[2] * 2.355 if fit is not None else None  # σ → FWHM

        results.append({
            '2θ (°)': round(two_theta_val, 2),
            'd-spacing (Å)': round(d, 4),
            'Rel. Intensity (%)': round(rel_intensity, 1),
            'FWHM (°)': round(fwhm, 4) if fwhm else 'N/A',
        })

    return pd.DataFrame(results), peaks_idx

# ── Publication-quality plot ──────────────────────────────────────────────────
def plot_xrd(two_theta, intensity, df_results, peaks_idx, material='α-Fe (BCC)'):
    fig = plt.figure(figsize=(12, 8), facecolor='#0f1117')
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)

    # ── Main XRD pattern ──
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#0f1117')

    ax1.fill_between(two_theta, intensity, alpha=0.15, color='#4fc3f7')
    ax1.plot(two_theta, intensity, color='#4fc3f7', linewidth=1.2, label='XRD Pattern')

    colors = ['#ff6b6b', '#ffd93d', '#6bcb77', '#c77dff']
    hkl_labels = ['(110)', '(200)', '(211)', '(220)']

    for i, (idx, row) in enumerate(zip(peaks_idx, df_results.iterrows())):
        _, row_data = row
        c = colors[i % len(colors)]
        label = hkl_labels[i] if i < len(hkl_labels) else f'P{i+1}'
        ax1.axvline(x=two_theta[idx], color=c, linestyle='--', alpha=0.6, linewidth=0.9)
        ax1.annotate(
            f"{label}\n{row_data['2θ (°)']}°\nd={row_data['d-spacing (Å)']}Å",
            xy=(two_theta[idx], intensity[idx]),
            xytext=(two_theta[idx] + 1.5, intensity[idx] - 0.08),
            fontsize=8.5, color=c,
            arrowprops=dict(arrowstyle='->', color=c, lw=0.8),
        )
        ax1.plot(two_theta[idx], intensity[idx], 'o', color=c, markersize=6, zorder=5)

    ax1.set_ylabel('Normalized Intensity (a.u.)', color='#cccccc', fontsize=11)
    ax1.set_title(f'XRD Pattern Analysis — {material}  |  Cu Kα λ = 1.5406 Å',
                  color='white', fontsize=13, pad=14, fontweight='bold')
    ax1.tick_params(colors='#888888', labelbottom=False)
    for spine in ax1.spines.values():
        spine.set_edgecolor('#333333')
    ax1.set_xlim(two_theta[0], two_theta[-1])
    ax1.set_ylim(-0.02, 1.12)
    ax1.legend(facecolor='#1a1a2e', edgecolor='#333333', labelcolor='white', fontsize=9)
    ax1.grid(axis='x', color='#222222', linestyle=':', linewidth=0.5)

    # ── Results table ──
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor('#0f1117')
    ax2.set_axis_off()

    col_labels = ['2θ (°)', 'd-spacing (Å)', 'Rel. Intensity (%)', 'FWHM (°)']
    table_data = [[row['2θ (°)'], row['d-spacing (Å)'],
                   row['Rel. Intensity (%)'], row['FWHM (°)']]
                  for _, row in df_results.iterrows()]

    table = ax2.table(cellText=table_data, colLabels=col_labels,
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#333333')
        if r == 0:
            cell.set_facecolor('#1a1a2e')
            cell.set_text_props(color='#4fc3f7', fontweight='bold')
        else:
            cell.set_facecolor('#161b22')
            cell.set_text_props(color='#cccccc')

    fig.text(0.99, 0.01, 'github.com/AashrithaNarala | NC State MSE',
             ha='right', va='bottom', fontsize=7.5, color='#444444', style='italic')

    plt.savefig('xrd_analysis_output.png', dpi=180, bbox_inches='tight',
                facecolor='#0f1117')
    plt.savefig('xrd_analysis_output.pdf', bbox_inches='tight',
                facecolor='#0f1117')
    print("Saved: xrd_analysis_output.png and .pdf")
    plt.show()

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating α-Fe XRD pattern...")
    two_theta, intensity = generate_iron_xrd(noise_level=0.025)

    print("Detecting peaks and calculating d-spacings...")
    df_results, peaks_idx = analyze_xrd(two_theta, intensity)

    print("\n── Peak Analysis Results ──────────────────────")
    print(df_results.to_string(index=False))
    print("───────────────────────────────────────────────\n")

    plot_xrd(two_theta, intensity, df_results, peaks_idx)
    df_results.to_csv('xrd_peak_data.csv', index=False)
    print("Data exported to xrd_peak_data.csv")
