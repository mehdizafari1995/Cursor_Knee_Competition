# Literature Search Log

Pilot scope: records returned by the documented queries on the execution dates; no claim of exhaustiveness is made.

| Query | Database | Executed UTC | Hits | Retained |
| --- | --- | --- | ---: | ---: |
| `Q01_ligaments` | pubmed | 2026-09-13T23:26:34.157740+00:00 | 2891 | 8 |
| `Q02_menisci` | pubmed | 2026-09-13T23:26:35.714048+00:00 | 4354 | 8 |
| `Q03_osteoarthritis` | pubmed | 2026-09-13T23:26:37.709235+00:00 | 1386 | 8 |
| `Q04_fluid_synovium_cyst` | pubmed | 2026-09-13T23:26:39.576897+00:00 | 1362 | 8 |
| `Q05_bone_trauma` | pubmed | 2026-09-13T23:26:41.674622+00:00 | 1196 | 8 |
| `Q06_sequences_planes_terminology` | pubmed | 2026-09-13T23:26:43.702145+00:00 | 1963 | 8 |
| `Q07_report_language` | pubmed | 2026-09-13T23:26:45.493802+00:00 | 868 | 5 |
| `Q08_machine_learning` | pubmed | 2026-09-13T23:26:47.177429+00:00 | 385 | 5 |
| `Q09_report_weak_supervision` | pubmed | 2026-09-13T23:26:48.940511+00:00 | 6 | 3 |
| `Q10_multi_instance` | pubmed | 2026-09-13T23:26:50.508048+00:00 | 1 | 1 |
| `Q11_2_5d` | pubmed | 2026-09-13T23:26:51.840820+00:00 | 2 | 2 |

Combined unique records after PMID → DOI → normalized-title deduplication: 59.

## Q01_ligaments

Question: Which PubMed records characterize ACL and MCL injury on knee MRI, including diagnostic terminology and mimics?

Exact query: `("Knee Joint"[MeSH Terms] OR knee[Title/Abstract]) AND ("Magnetic Resonance Imaging"[MeSH Terms] OR MRI[Title/Abstract] OR "MR imaging"[Title/Abstract]) AND (("Anterior Cruciate Ligament Injuries"[MeSH Terms] OR ACL[Title/Abstract] OR "anterior cruciate ligament"[Title/Abstract] OR "Medial Collateral Ligament"[MeSH Terms] OR MCL[Title/Abstract] OR "medial collateral ligament"[Title/Abstract]) AND (injur*[Title/Abstract] OR tear*[Title/Abstract] OR sprain*[Title/Abstract] OR diagnos*[Title/Abstract] OR mimic*[Title/Abstract]))`

PubMed translation: `("Knee Joint"[MeSH Terms] OR "knee"[Title/Abstract]) AND ("Magnetic Resonance Imaging"[MeSH Terms] OR "MRI"[Title/Abstract] OR "MR imaging"[Title/Abstract]) AND (("Anterior Cruciate Ligament Injuries"[MeSH Terms] OR "ACL"[Title/Abstract] OR "anterior cruciate ligament"[Title/Abstract] OR "MCL"[Title/Abstract] OR "medial collateral ligament"[Title/Abstract]) AND ("injur*"[Title/Abstract] OR "tear*"[Title/Abstract] OR "sprain*"[Title/Abstract] OR "diagnos*"[Title/Abstract] OR "mimic*"[Title/Abstract]))`

Limits: `{"date_range": null, "language": null, "publication_type": null, "result_order": "relevance", "per_query_retention_cap": 8}`

Screening: Retain up to the first 8 PubMed records sorted by relevance that returned parseable efetch metadata; global deduplication is PMID, then DOI, then normalized title.

## Q02_menisci

Question: Which PubMed records characterize medial and lateral meniscal injury on knee MRI, including tear terminology and mimics?

Exact query: `("Knee Joint"[MeSH Terms] OR knee[Title/Abstract]) AND ("Magnetic Resonance Imaging"[MeSH Terms] OR MRI[Title/Abstract] OR "MR imaging"[Title/Abstract]) AND (("Tibial Meniscus Injuries"[MeSH Terms] OR menisc*[Title/Abstract] OR "medial meniscus"[Title/Abstract] OR "lateral meniscus"[Title/Abstract]) AND (injur*[Title/Abstract] OR tear*[Title/Abstract] OR diagnos*[Title/Abstract] OR sign*[Title/Abstract] OR mimic*[Title/Abstract]))`

PubMed translation: `("Knee Joint"[MeSH Terms] OR "knee"[Title/Abstract]) AND ("Magnetic Resonance Imaging"[MeSH Terms] OR "MRI"[Title/Abstract] OR "MR imaging"[Title/Abstract]) AND (("Tibial Meniscus Injuries"[MeSH Terms] OR "menisc*"[Title/Abstract] OR "medial meniscus"[Title/Abstract] OR "lateral meniscus"[Title/Abstract]) AND ("injur*"[Title/Abstract] OR "tear*"[Title/Abstract] OR "diagnos*"[Title/Abstract] OR "sign*"[Title/Abstract] OR "mimic*"[Title/Abstract]))`

Limits: `{"date_range": null, "language": null, "publication_type": null, "result_order": "relevance", "per_query_retention_cap": 8}`

Screening: Retain up to the first 8 PubMed records sorted by relevance that returned parseable efetch metadata; global deduplication is PMID, then DOI, then normalized title.

## Q03_osteoarthritis

Question: Which PubMed records describe medial, lateral, and patellofemoral compartment osteoarthritis findings on knee MRI?

Exact query: `("Knee Joint"[MeSH Terms] OR knee[Title/Abstract]) AND ("Magnetic Resonance Imaging"[MeSH Terms] OR MRI[Title/Abstract] OR "MR imaging"[Title/Abstract]) AND (("Osteoarthritis, Knee"[MeSH Terms] OR osteoarthrit*[Title/Abstract] OR OA[Title/Abstract]) AND (medial[Title/Abstract] OR lateral[Title/Abstract] OR patellofemoral[Title/Abstract] OR compartment*[Title/Abstract]) AND (finding*[Title/Abstract] OR diagnos*[Title/Abstract] OR grad*[Title/Abstract] OR feature*[Title/Abstract]))`

PubMed translation: `("Knee Joint"[MeSH Terms] OR "knee"[Title/Abstract]) AND ("Magnetic Resonance Imaging"[MeSH Terms] OR "MRI"[Title/Abstract] OR "MR imaging"[Title/Abstract]) AND (("osteoarthritis, knee"[MeSH Terms] OR "osteoarthrit*"[Title/Abstract] OR "OA"[Title/Abstract]) AND ("medial"[Title/Abstract] OR "lateral"[Title/Abstract] OR "patellofemoral"[Title/Abstract] OR "compartment*"[Title/Abstract]) AND ("finding*"[Title/Abstract] OR "diagnos*"[Title/Abstract] OR "grad*"[Title/Abstract] OR "feature*"[Title/Abstract]))`

Limits: `{"date_range": null, "language": null, "publication_type": null, "result_order": "relevance", "per_query_retention_cap": 8}`

Screening: Retain up to the first 8 PubMed records sorted by relevance that returned parseable efetch metadata; global deduplication is PMID, then DOI, then normalized title.

## Q04_fluid_synovium_cyst

Question: Which PubMed records characterize knee joint effusion, synovitis, and Baker cyst on MRI?

Exact query: `("Knee Joint"[MeSH Terms] OR knee[Title/Abstract]) AND ("Magnetic Resonance Imaging"[MeSH Terms] OR MRI[Title/Abstract] OR "MR imaging"[Title/Abstract]) AND (("Joint Effusion"[MeSH Terms] OR effusion*[Title/Abstract] OR "Synovitis"[MeSH Terms] OR synovitis[Title/Abstract] OR "Popliteal Cyst"[MeSH Terms] OR "Baker cyst"[Title/Abstract] OR "Baker's cyst"[Title/Abstract] OR "popliteal cyst"[Title/Abstract]) AND (finding*[Title/Abstract] OR diagnos*[Title/Abstract] OR assess*[Title/Abstract] OR mimic*[Title/Abstract]))`

PubMed translation: `("Knee Joint"[MeSH Terms] OR "knee"[Title/Abstract]) AND ("Magnetic Resonance Imaging"[MeSH Terms] OR "MRI"[Title/Abstract] OR "MR imaging"[Title/Abstract]) AND (("effusion*"[Title/Abstract] OR "Synovitis"[MeSH Terms] OR "Synovitis"[Title/Abstract] OR "Popliteal Cyst"[MeSH Terms] OR "Baker cyst"[Title/Abstract] OR "Baker's cyst"[Title/Abstract] OR "Popliteal Cyst"[Title/Abstract]) AND ("finding*"[Title/Abstract] OR "diagnos*"[Title/Abstract] OR "assess*"[Title/Abstract] OR "mimic*"[Title/Abstract]))`

Limits: `{"date_range": null, "language": null, "publication_type": null, "result_order": "relevance", "per_query_retention_cap": 8}`

Screening: Retain up to the first 8 PubMed records sorted by relevance that returned parseable efetch metadata; global deduplication is PMID, then DOI, then normalized title.

## Q05_bone_trauma

Question: Which PubMed records characterize knee bone contusion and fracture on MRI, including occult injury and mimics?

Exact query: `("Knee Joint"[MeSH Terms] OR knee[Title/Abstract]) AND ("Magnetic Resonance Imaging"[MeSH Terms] OR MRI[Title/Abstract] OR "MR imaging"[Title/Abstract]) AND (("Bone and Bones/injuries"[MeSH Terms] OR "bone contusion"[Title/Abstract] OR "bone bruise"[Title/Abstract] OR "bone marrow edema"[Title/Abstract] OR fracture*[Title/Abstract] OR "occult fracture"[Title/Abstract]) AND (diagnos*[Title/Abstract] OR finding*[Title/Abstract] OR trauma*[Title/Abstract] OR occult[Title/Abstract] OR mimic*[Title/Abstract]))`

PubMed translation: `("Knee Joint"[MeSH Terms] OR "knee"[Title/Abstract]) AND ("Magnetic Resonance Imaging"[MeSH Terms] OR "MRI"[Title/Abstract] OR "MR imaging"[Title/Abstract]) AND (("bone and bones/injuries"[MeSH Terms] OR "bone contusion"[Title/Abstract] OR "bone bruise"[Title/Abstract] OR "bone marrow edema"[Title/Abstract] OR "fracture*"[Title/Abstract] OR "occult fracture"[Title/Abstract]) AND ("diagnos*"[Title/Abstract] OR "finding*"[Title/Abstract] OR "trauma*"[Title/Abstract] OR "occult"[Title/Abstract] OR "mimic*"[Title/Abstract]))`

Limits: `{"date_range": null, "language": null, "publication_type": null, "result_order": "relevance", "per_query_retention_cap": 8}`

Screening: Retain up to the first 8 PubMed records sorted by relevance that returned parseable efetch metadata; global deduplication is PMID, then DOI, then normalized title.

## Q06_sequences_planes_terminology

Question: Which PubMed records address knee-MRI planes, sequences, finding terminology, abbreviations, or diagnostic mimics?

Exact query: `("Knee Joint"[MeSH Terms] OR knee[Title/Abstract]) AND ("Magnetic Resonance Imaging"[MeSH Terms] OR MRI[Title/Abstract] OR "MR imaging"[Title/Abstract]) AND ((sequence*[Title/Abstract] OR protocol*[Title/Abstract] OR sagittal[Title/Abstract] OR coronal[Title/Abstract] OR axial[Title/Abstract] OR "proton density"[Title/Abstract] OR "fat suppress*"[Title/Abstract] OR T1[Title/Abstract] OR T2[Title/Abstract]) AND (finding*[Title/Abstract] OR terminolog*[Title/Abstract] OR report*[Title/Abstract] OR abbreviation*[Title/Abstract] OR mimic*[Title/Abstract]))`

PubMed translation: `("Knee Joint"[MeSH Terms] OR "knee"[Title/Abstract]) AND ("Magnetic Resonance Imaging"[MeSH Terms] OR "MRI"[Title/Abstract] OR "MR imaging"[Title/Abstract]) AND (("sequence*"[Title/Abstract] OR "protocol*"[Title/Abstract] OR "sagittal"[Title/Abstract] OR "coronal"[Title/Abstract] OR "axial"[Title/Abstract] OR "proton density"[Title/Abstract] OR "fat suppress*"[Title/Abstract] OR "T1"[Title/Abstract] OR "T2"[Title/Abstract]) AND ("finding*"[Title/Abstract] OR "terminolog*"[Title/Abstract] OR "report*"[Title/Abstract] OR "abbreviation*"[Title/Abstract] OR "mimic*"[Title/Abstract]))`

Limits: `{"date_range": null, "language": null, "publication_type": null, "result_order": "relevance", "per_query_retention_cap": 8}`

Screening: Retain up to the first 8 PubMed records sorted by relevance that returned parseable efetch metadata; global deduplication is PMID, then DOI, then normalized title.

## Q07_report_language

Question: Which PubMed records address negation, uncertainty, postoperative or multilingual language, and weak supervision in radiology reports?

Exact query: `("Knee Joint"[MeSH Terms] OR knee[Title/Abstract]) AND ("Magnetic Resonance Imaging"[MeSH Terms] OR MRI[Title/Abstract] OR "MR imaging"[Title/Abstract]) AND (("Radiology Report"[Title/Abstract] OR "radiology reports"[Title/Abstract] OR report*[Title/Abstract]) AND (negation[Title/Abstract] OR uncertain*[Title/Abstract] OR equivocal[Title/Abstract] OR postop*[Title/Abstract] OR postsurg*[Title/Abstract] OR multilingual[Title/Abstract] OR synonym*[Title/Abstract] OR abbreviation*[Title/Abstract] OR "weak supervision"[Title/Abstract] OR "natural language processing"[Title/Abstract] OR NLP[Title/Abstract]))`

PubMed translation: `("Knee Joint"[MeSH Terms] OR "knee"[Title/Abstract]) AND ("Magnetic Resonance Imaging"[MeSH Terms] OR "MRI"[Title/Abstract] OR "MR imaging"[Title/Abstract]) AND (("Radiology Report"[Title/Abstract] OR "radiology reports"[Title/Abstract] OR "report*"[Title/Abstract]) AND ("negation"[Title/Abstract] OR "uncertain*"[Title/Abstract] OR "equivocal"[Title/Abstract] OR "postop*"[Title/Abstract] OR "postsurg*"[Title/Abstract] OR "multilingual"[Title/Abstract] OR "synonym*"[Title/Abstract] OR "abbreviation*"[Title/Abstract] OR "weak supervision"[Title/Abstract] OR "natural language processing"[Title/Abstract] OR "NLP"[Title/Abstract]))`

Limits: `{"date_range": null, "language": null, "publication_type": null, "result_order": "relevance", "per_query_retention_cap": 5}`

Screening: Retain up to the first 5 PubMed records sorted by relevance that returned parseable efetch metadata; global deduplication is PMID, then DOI, then normalized title.

## Q08_machine_learning

Question: Which PubMed records address knee-MRI classification using machine learning or deep learning?

Exact query: `("Knee Joint"[MeSH Terms] OR knee[Title/Abstract]) AND ("Magnetic Resonance Imaging"[MeSH Terms] OR MRI[Title/Abstract] OR "MR imaging"[Title/Abstract]) AND (("Machine Learning"[MeSH Terms] OR "deep learning"[Title/Abstract] OR "machine learning"[Title/Abstract] OR "artificial intelligence"[Title/Abstract]) AND (classific*[Title/Abstract] OR detect*[Title/Abstract] OR diagnos*[Title/Abstract]))`

PubMed translation: `("Knee Joint"[MeSH Terms] OR "knee"[Title/Abstract]) AND ("Magnetic Resonance Imaging"[MeSH Terms] OR "MRI"[Title/Abstract] OR "MR imaging"[Title/Abstract]) AND (("Machine Learning"[MeSH Terms] OR "deep learning"[Title/Abstract] OR "Machine Learning"[Title/Abstract] OR "artificial intelligence"[Title/Abstract]) AND ("classific*"[Title/Abstract] OR "detect*"[Title/Abstract] OR "diagnos*"[Title/Abstract]))`

Limits: `{"date_range": null, "language": null, "publication_type": null, "result_order": "relevance", "per_query_retention_cap": 5}`

Screening: Retain up to the first 5 PubMed records sorted by relevance that returned parseable efetch metadata; global deduplication is PMID, then DOI, then normalized title.

## Q09_report_weak_supervision

Question: Which PubMed records address report-derived or weakly supervised labels for knee-MRI machine learning?

Exact query: `("Knee Joint"[MeSH Terms] OR knee[Title/Abstract]) AND ("Magnetic Resonance Imaging"[MeSH Terms] OR MRI[Title/Abstract] OR "MR imaging"[Title/Abstract]) AND (("radiology report"[Title/Abstract] OR "radiology reports"[Title/Abstract] OR "report-derived"[Title/Abstract] OR "report derived"[Title/Abstract]) AND ("weak supervision"[Title/Abstract] OR "weakly supervised"[Title/Abstract] OR "natural language processing"[Title/Abstract] OR NLP[Title/Abstract] OR label*[Title/Abstract]))`

PubMed translation: `("Knee Joint"[MeSH Terms] OR "knee"[Title/Abstract]) AND ("Magnetic Resonance Imaging"[MeSH Terms] OR "MRI"[Title/Abstract] OR "MR imaging"[Title/Abstract]) AND (("radiology report"[Title/Abstract] OR "radiology reports"[Title/Abstract] OR "report-derived"[Title/Abstract] OR "report-derived"[Title/Abstract]) AND ("weak supervision"[Title/Abstract] OR "weakly supervised"[Title/Abstract] OR "natural language processing"[Title/Abstract] OR "NLP"[Title/Abstract] OR "label*"[Title/Abstract]))`

Limits: `{"date_range": null, "language": null, "publication_type": null, "result_order": "relevance", "per_query_retention_cap": 3}`

Screening: Retain up to the first 3 PubMed records sorted by relevance that returned parseable efetch metadata; global deduplication is PMID, then DOI, then normalized title.

## Q10_multi_instance

Question: Which PubMed records apply multi-instance or multiple-instance learning to knee MRI?

Exact query: `("Knee Joint"[MeSH Terms] OR knee[Title/Abstract]) AND ("Magnetic Resonance Imaging"[MeSH Terms] OR MRI[Title/Abstract] OR "MR imaging"[Title/Abstract]) AND ("multi-instance learning"[Title/Abstract] OR "multiple-instance learning"[Title/Abstract] OR "multiple instance learning"[Title/Abstract])`

PubMed translation: `("Knee Joint"[MeSH Terms] OR "knee"[Title/Abstract]) AND ("Magnetic Resonance Imaging"[MeSH Terms] OR "MRI"[Title/Abstract] OR "MR imaging"[Title/Abstract]) AND ("multi-instance learning"[Title/Abstract] OR "multiple-instance learning"[Title/Abstract] OR "multiple-instance learning"[Title/Abstract])`

Limits: `{"date_range": null, "language": null, "publication_type": null, "result_order": "relevance", "per_query_retention_cap": 3}`

Screening: Retain up to the first 3 PubMed records sorted by relevance that returned parseable efetch metadata; global deduplication is PMID, then DOI, then normalized title.

## Q11_2_5d

Question: Which PubMed records apply 2.5D classification methods to knee MRI?

Exact query: `("Knee Joint"[MeSH Terms] OR knee[Title/Abstract]) AND ("Magnetic Resonance Imaging"[MeSH Terms] OR MRI[Title/Abstract] OR "MR imaging"[Title/Abstract]) AND ("2.5D"[Title/Abstract] OR "2·5D"[Title/Abstract] OR "two-and-a-half-dimensional"[Title/Abstract]) AND (classific*[Title/Abstract] OR detect*[Title/Abstract] OR diagnos*[Title/Abstract] OR "deep learning"[Title/Abstract])`

PubMed translation: `("Knee Joint"[MeSH Terms] OR "knee"[Title/Abstract]) AND ("Magnetic Resonance Imaging"[MeSH Terms] OR "MRI"[Title/Abstract] OR "MR imaging"[Title/Abstract]) AND ("2.5D"[Title/Abstract] OR "2.5D"[Title/Abstract] OR "two-and-a-half-dimensional"[Title/Abstract]) AND ("classific*"[Title/Abstract] OR "detect*"[Title/Abstract] OR "diagnos*"[Title/Abstract] OR "deep learning"[Title/Abstract])`

Limits: `{"date_range": null, "language": null, "publication_type": null, "result_order": "relevance", "per_query_retention_cap": 3}`

Screening: Retain up to the first 3 PubMed records sorted by relevance that returned parseable efetch metadata; global deduplication is PMID, then DOI, then normalized title.

