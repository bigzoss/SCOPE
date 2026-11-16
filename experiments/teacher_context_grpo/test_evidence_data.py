import json
import unittest

from evidence_data import (build_examples, category, diagnostic_views, evidence_messages,
                           extend_view, inventory, parse_proposal, format_signals)
from evidence_teacher import parse_batch


class EvidenceContractTests(unittest.TestCase):
    def setUp(self):
        self.public = {'id':'train-1','question':'Where was the founder of A born?',
                       'paragraphs': [('A',['A was founded by B.']),
                                      ('B',['B was born in C.']),
                                      ('D',['D was founded by E.'])]}
        self.gold = {'answer':'C','supporting_facts':[('A',0),('B',0)]}
        self.view = extend_view([],self.public,['D0:S0'])

    def test_teacher_cannot_cite_missing_source(self):
        with self.assertRaises(ValueError):
            parse_proposal('{"answer":"C","evidence":["D8:S0"]}',self.public)

    def test_batched_teacher_is_bound_to_each_task(self):
        other={'id':'train-2','question':'Where?','paragraphs':[('X',['X is here.'])]}
        bad={'results':[{'task_id':'train-1','answer':'C','evidence':['D0:S0']},
                        {'task_id':'train-2','answer':'here','evidence':['D1:S0']}]}
        with self.assertRaises(ValueError):parse_batch(json.dumps(bad),[self.public,other])
        bad['results'][1]['evidence']=['D0:S0']
        self.assertEqual(set(parse_batch(json.dumps(bad),[self.public,other])),{'train-1','train-2'})
        bad['results'][1]['task_id']='train-1'
        with self.assertRaises(ValueError):parse_batch(json.dumps(bad),[self.public,other])

    def test_missing_support_rejected_even_with_correct_answer(self):
        rows,why=build_examples(self.public,self.gold,{'answer':'unknown','answer_view':self.view},
                                {'answer':'C','evidence':['D0:S0']})
        self.assertEqual(rows,[])
        self.assertEqual(why,'teacher_citations_incomplete')

    def test_teacher_inputs_have_no_hidden_labels(self):
        content=json.loads(evidence_messages(self.public)[-1]['content'])
        self.assertEqual(set(content),{'question','sentences'})
        self.assertEqual(content['sentences'],inventory(self.public))

    def test_examples_separate_citation_supervision(self):
        rows,why=build_examples(self.public,self.gold,{'answer':'unknown','answer_view':self.view},
                                {'answer':'C','evidence':['D0:S0','D1:S0']})
        self.assertIsNone(why)
        self.assertEqual(rows[0]['completion'],'C')
        self.assertEqual(rows[0]['diagnostic_category'],'support_missing')
        self.assertEqual(json.loads(rows[1]['completion'])['evidence'],['D0:S0','D1:S0'])

    def test_distractor_excludes_all_support_documents(self):
        views,meta=diagnostic_views(self.public,self.gold,self.view,lambda t:len(t.split()))
        self.assertEqual(meta['missing_support_sentences'],1)
        self.assertEqual({x['title'] for x in views['support_augmented']},{'A','B'})
        self.assertNotIn('B',{x['title'] for x in views['distractor_augmented']})
        self.assertEqual(views['native'],self.view)

    def test_format_is_only_a_candidate_and_requires_visible_support(self):
        full=extend_view(self.view,self.public,['D1:S0'])
        self.assertEqual(category('The answer is C.',full,self.gold),'format_candidate')
        self.assertEqual(category('The answer is C.',self.view,self.gold),'support_missing')
        self.assertEqual(category('yes, but no',full,{'answer':'yes','supporting_facts':[('A',0)]}),
                         'support_present_other_error')

    def test_posthoc_format_checks_do_not_claim_semantic_correctness(self):
        self.assertTrue(format_signals('Band-e-Amir National Park','Band-e Amir National Park')['punctuation_spacing_equivalent'])
        self.assertTrue(format_signals('No. The locations differ.','No')['binary_prefix_matches'])
        self.assertFalse(format_signals('Unknown','North Holland')['format_compatible'])
        self.assertTrue(format_signals('Yes, but possibly not.','Yes')['requires_semantic_review'])


if __name__=='__main__':
    unittest.main()
