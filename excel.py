import xlsxwriter

workbook = xlsxwriter.Workbook(
    'RAG Algorithm Test.xlsx',
)
worksheet = workbook.add_worksheet(name='CCR')
bold = workbook.add_format({'bold': True})

# worksheet.write(0, 0, 'ID', bold)
# worksheet.write(0, 1, 'Question Code', bold)
# worksheet.write(0, 2, 'Method', bold)
worksheet.write(1, 0, '1')
worksheet.write(1, 1, 'Code')
worksheet.write(2, 2, 'vector')

workbook.close()