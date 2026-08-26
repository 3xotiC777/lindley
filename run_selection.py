import sys
import time
import config
import ingest
import selector_engine
import exporter

def main():
    print("==========================================================")
    print(f"       LINDLEY SAMPLE SELECTION ENGINE - {config.MONTH_NAME}")
    print("==========================================================")
    start_time = time.time()
    
    # Step 1: Ingest & merge datasets (Partition ELEGIBLE vs NO ELEGIBLE)
    df_elegible, df_no_elegible, historical = ingest.prepare_merged_datasets()
    
    # Step 2: Run selection rules strictly on ELEGIBLE universe for BD sheet
    final_elegible, controls = selector_engine.run_selection_process(df_elegible, historical)
    
    # Step 3: Export output workbooks
    exporter.export_selection_workbook(final_elegible, df_no_elegible, controls, config.OUTPUT_SELECCION_PATH)
    exporter.export_def_sup_workbook(final_elegible, config.OUTPUT_DEF_SUP_PATH)
    
    elapsed = time.time() - start_time
    print("==========================================================")
    print(f" SUCCESS! Process completed in {elapsed:.2f} seconds.")
    print(f" Main Selection Output: {config.OUTPUT_SELECCION_PATH}")
    print(f" Archivo Def Sup Output: {config.OUTPUT_DEF_SUP_PATH}")
    print("==========================================================")

if __name__ == '__main__':
    main()
