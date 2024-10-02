from __future__ import annotations

import json

import xlsxwriter

from test_utils import questions, test_configs, get_test_results, DOCS_DIR, TestResult

workbook = xlsxwriter.Workbook(
    'RAG Algorithm Test.xlsx',
)
bold = workbook.add_format({'bold': True})

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
worksheet.write(0, 0, 'ID', bold)
worksheet.write(0, 1, 'Params', bold)

for i, conf in enumerate(test_configs.configs, 1):
    worksheet.write(i, 0, conf.id)
    worksheet.write(i, 1, json.dumps(conf.params))

worksheet.autofit()

# Document results sheets

for doc_name in ['CCR']:
    worksheet = workbook.add_worksheet(name=doc_name)

    worksheet.write(0, 0, 'ID', bold)
    worksheet.write(0, 1, 'Question Code', bold)

    for i, question in enumerate(questions, 1):
        worksheet.write(i, 0, i)
        worksheet.write(i, 1, question.Key)

    results: list[TestResult] = get_test_results(f'{DOCS_DIR}/{doc_name}.results')

    for i, result in enumerate(results, 2):
        worksheet.write(0, i, f'conf_id={result.conf_id}, chunk_size={result.chunk_size}', bold)

    worksheet.autofit()

    for i, result in enumerate(results, 2):
        for j, answer in enumerate(result.answers, 1):
            worksheet.write(j, i, answer)

workbook.close()
