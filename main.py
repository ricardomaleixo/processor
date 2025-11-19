import os
from processor import get_shape_properties

def process_all_steps(input_folder: str, output_csv: str | None = None):
    """
    Percorre todos os arquivos .stp/.step de uma pasta, processa com get_shape_properties
    e mostra (e opcionalmente salva em CSV).
    """
    step_exts = {".stp", ".step", ".STEP", ".STP"}

    files = [
        f for f in os.listdir(input_folder)
        if os.path.splitext(f)[1] in step_exts
    ]

    if not files:
        print(f"Nenhum arquivo STEP encontrado em: {input_folder}")
        return

    results = []

    for fname in files:
        fpath = os.path.join(input_folder, fname)
        print(f"🔧 Processando: {fname} ...")
        try:
            data = get_shape_properties(fpath)
            results.append(data)
            # Mostra um resumo no terminal
            print(f"  → Part: {data['Part']}")
            print(f"  → Material: {data['Material']}")
            print(f"  → Thickness: {data['Thickness']}")
            print(f"  → Cutting length: {data['Cutting length']}")
            print(f"  → Number of bends: {data['Number of bends']}")
            print(f"  → Mass (kg): {data['Mass']}")
            print()
        except Exception as e:
            print(f"  ERRO ao processar {fname}: {e}")

    # Salvar em CSV, se pedido
    if output_csv and results:
        import csv

        fieldnames = list(results[0].keys())
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
            writer.writeheader()
            for row in results:
                writer.writerow(row)

        print(f"✅ Resultados salvos em: {output_csv}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(base_dir, "stp_files")

    # Se quiser salvar em CSV:
    output_csv = os.path.join(base_dir, "resultado_step.csv")

    process_all_steps(input_folder, output_csv=output_csv)
