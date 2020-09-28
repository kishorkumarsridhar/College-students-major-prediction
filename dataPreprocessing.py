import csv

def parse_csv(input_file, output_file):
    stud_info = {}
    subject_list = set()
    id_name = 'id'

    with open(input_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            student_id = row[0]
            subject_name = row[1]
            #major = row[2] # Used only for training data
            if line_count == 0:
                line_count += 1
                pass
            else:
                if student_id in stud_info:
                    if subject_name in stud_info[student_id]:
                        stud_info[student_id][subject_name] += 1
                    else:
                        stud_info[student_id][subject_name] = 1
                else:
                    stud_info[student_id] = {}
                    stud_info[student_id][id_name] = student_id
                    stud_info[student_id][subject_name] = 1 
                    #stud_info[student_id]['major'] = major # Used only for training data
                line_count += 1
                subject_list.add(subject_name)

    with open(output_file, mode='w') as csv_file:
       # writer = csv.DictWriter(csv_file, delimiter=',', restval=0, fieldnames=[id_name]+list(subject_list)+['major']) # Used only for training data 
        writer = csv.DictWriter(csv_file, delimiter=',', restval=0, fieldnames=[id_name]+list(subject_list))
        writer.writeheader()
        for student in stud_info:
            writer.writerow(stud_info[student])

if __name__ == '__main__':
    input_file = 'training.csv'
    output_file = 'training_new.csv'
    parse_csv(input_file, output_file)
    
 if __name__ == '__main__':
    input_file = 'eval.csv'
    output_file = 'eval_new.csv'
    parse_csv(input_file, output_file)