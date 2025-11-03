"""
Configuration Module
Contains feature descriptions and app settings
"""

FEATURE_DESCRIPTIONS = {
    'surgery': {
        'name': 'Surgery Status',
        'description': 'Whether the horse had surgery (1=Yes, 2=No)',
        'type': 'categorical'
    },
    'age': {
        'name': 'Age Category',
        'description': 'Age of horse (1=Adult, 2=Young <6 months)',
        'type': 'categorical'
    },
    'rectal_temp': {
        'name': 'Rectal Temperature',
        'description': 'Temperature in Celsius (normal: 37.8°C). Elevated temp may indicate infection',
        'type': 'numerical'
    },
    'pulse': {
        'name': 'Heart Rate',
        'description': 'Heart rate in beats/minute (normal: 30-40 for adults). Elevated in painful lesions or shock',
        'type': 'numerical'
    },
    'respiratory_rate': {
        'name': 'Respiratory Rate',
        'description': 'Breathing rate per minute (normal: 8-10)',
        'type': 'numerical'
    },
    'temp_of_extremities': {
        'name': 'Temperature of Extremities',
        'description': 'Peripheral circulation (1=Normal, 2=Warm, 3=Cool, 4=Cold). Cool/cold indicates shock',
        'type': 'categorical'
    },
    'peripheral_pulse': {
        'name': 'Peripheral Pulse',
        'description': 'Pulse quality (1=Normal, 2=Increased, 3=Reduced, 4=Absent). Reduced/absent indicates poor perfusion',
        'type': 'categorical'
    },
    'mucous_membrane': {
        'name': 'Mucous Membrane Color',
        'description': 'Color indicating circulation (1=Normal pink, 2=Bright pink, 3=Pale pink, 4=Pale cyanotic, 5=Bright red, 6=Dark cyanotic)',
        'type': 'categorical'
    },
    'capillary_refill_time': {
        'name': 'Capillary Refill Time',
        'description': 'Time for capillary refill (1=<3 sec, 2=>=3 sec). Longer refill indicates poorer circulation',
        'type': 'categorical'
    },
    'pain': {
        'name': 'Pain Level',
        'description': 'Subjective pain assessment (1=Alert no pain, 2=Depressed, 3=Intermittent mild, 4=Intermittent severe, 5=Continuous severe)',
        'type': 'categorical'
    },
    'peristalsis': {
        'name': 'Gut Activity (Peristalsis)',
        'description': 'Activity in horse gut (1=Hypermotile, 2=Normal, 3=Hypomotile, 4=Absent). Decreases as gut becomes distended',
        'type': 'categorical'
    },
    'abdominal_distension': {
        'name': 'Abdominal Distension',
        'description': 'Abdominal swelling level (1=None, 2=Slight, 3=Moderate, 4=Severe). Severe distension likely requires surgery',
        'type': 'categorical'
    },
    'nasogastric_tube': {
        'name': 'Nasogastric Tube Gas',
        'description': 'Gas from nasogastric tube (1=None, 2=Slight, 3=Significant). Large gas cap causes discomfort',
        'type': 'categorical'
    },
    'nasogastric_reflux': {
        'name': 'Nasogastric Reflux Amount',
        'description': 'Reflux amount (1=None, 2=>1 liter, 3=<1 liter). Greater reflux indicates obstruction',
        'type': 'categorical'
    },
    'nasogastric_reflux_ph': {
        'name': 'Nasogastric Reflux pH',
        'description': 'pH of reflux (scale 0-14, normal: 3-4, neutral: 7)',
        'type': 'numerical'
    },
    'rectal_exam_feces': {
        'name': 'Rectal Exam - Feces',
        'description': 'Feces assessment (1=Normal, 2=Increased, 3=Decreased, 4=Absent). Absent indicates obstruction',
        'type': 'categorical'
    },
    'abdomen': {
        'name': 'Abdominal Examination',
        'description': 'Findings (1=Normal, 2=Other, 3=Firm feces, 4=Distended small intestine, 5=Distended large intestine)',
        'type': 'categorical'
    },
    'packed_cell_volume': {
        'name': 'Packed Cell Volume',
        'description': 'Red blood cells by volume % (normal: 30-50). Rises with compromised circulation or dehydration',
        'type': 'numerical'
    },
    'total_protein': {
        'name': 'Total Protein',
        'description': 'Blood protein in gms/dL (normal: 6-7.5, default: 7.0). Higher values indicate dehydration',
        'type': 'numerical'
    },
    'abdomo_appearance': {
        'name': 'Abdominal Fluid Appearance',
        'description': 'Fluid from abdomen (1=Clear, 2=Cloudy, 3=Serosanguinous). Cloudy/serosanguinous indicates compromised gut',
        'type': 'categorical'
    },
    'abdomo_protein': {
        'name': 'Abdominal Fluid Protein',
        'description': 'Protein in abdominal fluid (gms/dL). Higher level indicates compromised gut',
        'type': 'numerical'
    },
    'surgical_lesion': {
        'name': 'Surgical Lesion',
        'description': 'Was problem surgical? (1=Yes, 2=No)',
        'type': 'categorical'
    },
    'lesion_1': {
        'name': 'Lesion Site',
        'description': 'Location of lesion (coded)',
        'type': 'numerical'
    },
    'lesion_2': {
        'name': 'Lesion Type',
        'description': 'Type of lesion (coded)',
        'type': 'numerical'
    },
    'lesion_3': {
        'name': 'Lesion Subtype',
        'description': 'Subtype of lesion (coded)',
        'type': 'numerical'
    },
    'cp_data': {
        'name': 'Pathology Data',
        'description': 'Whether pathology data present (1=Yes, 2=No)',
        'type': 'categorical'
    }
}

# Outcome class descriptions
OUTCOME_DESCRIPTIONS = {
    0: 'Lived',
    1: 'Died',
    2: 'Euthanized'
}

# App configuration
APP_CONFIG = {
    'title': 'Horse Survival Prediction System',
    'description': 'Predict horse survival based on medical conditions with explainable AI',
    'model_params': {
        'n_estimators': 200,
        'max_depth': 15,
        'random_state': 42
    }
}
