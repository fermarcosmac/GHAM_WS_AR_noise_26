# Final Monte Carlo identification tables

The two LaTeX fragments each contain PL and FSM tables. They require booktabs.
Copy the fragments into the manuscript or use \input at the desired locations.
The matching CSV files retain full precision, counts, statuses, and skipped methods.

Regenerate from the repository root in MATLAB:

    addpath('mlab');
    export_montecarlo_paper_tables();

The exporter reads comparison_pl_montecarlo.mat and comparison_fsm_montecarlo.mat
for example_1 and example_CSTR through plot_comparison_results. It runs no
identification or sampling and does not modify the saved MAT files.

Each reported method has 30 finite estimates. Tables use means and sample
standard deviations (n-1) for final relative parameter error (%), final output
RMSE, and complete wall-clock time (s). The same saved Python BI-HMC results
are reused in PL and FSM and placed last. BI-HMC timing includes warm-up and
sampling. All 30 BI-HMC fits in each example have sampling_warning status;
their finite point estimates are retained, as stated in the table captions.
All displayed means and standard deviations use four decimal places. Column-minimum
means are bold; ties are determined at the displayed precision. CSV precision is unchanged.
Only wholly skipped methods are omitted from the LaTeX tables.

For a four-table preview, run from this directory:

    pdflatex -output-directory=../pdf montecarlo_tables_preview.tex

The compiled preview is saved in output/pdf/montecarlo_tables_preview.pdf.
