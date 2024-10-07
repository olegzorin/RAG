from __future__ import annotations

import json

import xlsxwriter

from test_utils import questions, test_configs, get_test_results, DOCS_DIR, TestResult, doc_names

workbook = xlsxwriter.Workbook(
    'RAG Algorithm Test.xlsx',
)
bold = workbook.add_format({'bold': True})
center = workbook.add_format({'align': 'center'})
bold_center = workbook.add_format({'bold': True, 'align': 'center'})

# Questions sheet

worksheet = workbook.add_worksheet(name='Questions')
worksheet.write(0, 0, 'ID', bold)
worksheet.write(0, 1, 'Question Code', bold)
worksheet.write(0, 2, 'Question', bold)

for i, question in enumerate(questions, 1):
    worksheet.write(i, 0, i)
    worksheet.write(i, 1, question.Key)
    worksheet.write(i, 2, question.Value)

worksheet.autofit()

# Test Configs sheet

worksheet = workbook.add_worksheet(name='Test Configs')
worksheet.write(0, 0, 'Cnf ID', bold_center)
worksheet.write(0, 1, 'Params', bold)

for i, conf in enumerate(test_configs.configs, 1):
    worksheet.write(i, 0, conf.id.upper(), center)
    worksheet.write(i, 1, json.dumps(conf.params))

worksheet.autofit()

# Document results sheets

for doc_name in doc_names:
    results: list[TestResult] = get_test_results(f'{DOCS_DIR}/{doc_name}.results')

    # Question - Row

    worksheet = workbook.add_worksheet(name=doc_name)

    worksheet.write(0, 0, 'ID', bold)
    worksheet.write(0, 1, 'Question Code', bold)

    for i, question in enumerate(questions, 1):
        worksheet.write(i, 0, i)
        worksheet.write(i, 1, question.Key)

    for i, result in enumerate(results, 2):
        worksheet.write(0, i, f'Cnf ID:{result.conf_id.upper()}, ChunkSize:{result.chunk_size}', bold)

    worksheet.set_column(2, len(results) + 1, width=25.0)
    worksheet.autofit()

    for i, result in enumerate(results, 2):
        for j, answer in enumerate(result.answers, 1):
            worksheet.write(j, i, answer)

    # Question - Column

    worksheet_t = workbook.add_worksheet(name=f'{doc_name}-T')

    worksheet_t.write(1, 0, 'Cnf ID', bold_center)
    worksheet_t.write(1, 1, 'ChunkSize', bold_center)

    for i, result in enumerate(results, 2):
        worksheet_t.write(i, 0, result.conf_id.upper(), center)
        worksheet_t.write(i, 1, result.chunk_size, center)

    for i, question in enumerate(questions, 1):
        worksheet_t.write(0, i + 1, i, bold_center)
        worksheet_t.write(1, i + 1, question.Key, bold_center)

    worksheet_t.set_column(2, len(questions) + 1, width=25.0)
    worksheet_t.autofit()

    for i, result in enumerate(results, 2):
        for j, answer in enumerate(result.answers, 2):
            worksheet_t.write(i, j, answer)

workbook.close()
