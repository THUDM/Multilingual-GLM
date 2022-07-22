from tasks.data_utils import InputExample


class PVP:
    def __init__(self, tokenizer, max_src_length, max_tgt_length, task_mask=False):
        self.tokenizer = tokenizer
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.task_mask = task_mask

    @property
    def cls_id(self):
        return self.tokenizer.get_command('ENC').Id

    @property
    def mask_id(self):
        return self.tokenizer.get_command('MASK').Id

    def encode(self, example: InputExample):
        raise NotImplementedError



class NCLSPVP(PVP):
    @property
    def mask_id(self):
        mask_token = 'sMASK' if self.task_mask else 'MASK'
        return self.tokenizer.get_command(mask_token).Id

    def encode(self, example: InputExample):
        source_text, target_text = example.text_a, example.text_b
        source_tokens = self.tokenizer.EncodeAsIds(" " + source_text).tokenization
        prompt = [self.cls_id] #+ self.tokenizer.EncodeAsIds(" Content:").tokenization
        
        assert example.meta['task'] in ['sum', 'trans']
        if example.meta['task'] == 'sum':
            if example.meta['tgt_lang'] == 'en':
                prompt_task = self.tokenizer.EncodeAsIds(" TL;DR:").tokenization + [self.mask_id]
            elif example.meta['tgt_lang'] == 'zh':
                prompt_task = self.tokenizer.EncodeAsIds(" 中文摘要:").tokenization + [self.mask_id]
        else:
            prompt_task = self.tokenizer.EncodeAsIds(" 中文翻译:").tokenization + [self.mask_id]
        
        
        if len(source_tokens) > self.max_src_length - len(prompt)- len(prompt_task):
            source_tokens = source_tokens[:self.max_src_length - len(prompt) - len(prompt_task)]
        source_tokens = prompt + source_tokens + prompt_task
        #source_tokens = [self.cls_id] + source_tokens + prompt_tldr + [self.mask_id]
        return source_tokens, target_text



class SummaryPVP(PVP):
    @property
    def mask_id(self):
        mask_token = 'sMASK' if self.task_mask else 'MASK'
        return self.tokenizer.get_command(mask_token).Id

    def encode(self, example: InputExample):
        source_text, target_text = example.text_a, example.text_b
        source_tokens = self.tokenizer.EncodeAsIds(" " + source_text).tokenization
        prompt = [self.cls_id] #+ self.tokenizer.EncodeAsIds(" Content:").tokenization
        
        prompt_summary = self.tokenizer.EncodeAsIds(" tl;dr:").tokenization + [self.mask_id]  
        if 'tgt_lang' in example.meta:
            if example.meta['tgt_lang'] == 'zh':
                prompt_summary = self.tokenizer.EncodeAsIds(" 中文摘要:").tokenization + [self.mask_id]
        
        if len(source_tokens) > self.max_src_length - len(prompt)- len(prompt_summary):
            source_tokens = source_tokens[:self.max_src_length - len(prompt) - len(prompt_summary)]
        source_tokens = prompt + source_tokens + prompt_summary
        #source_tokens = [self.cls_id] + source_tokens + prompt_tldr + [self.mask_id]
        return source_tokens, target_text

class MLQAPVP(PVP): #Multilingual PVP for MLQA dataset
    
    @property
    def mask_id(self):
        mask_token = 'sMASK' if self.task_mask else 'MASK'
        return self.tokenizer.get_command(mask_token).Id
    
    
    def encode(self, example: InputExample):
        
        lang = example.meta['language'] #the language of the source and answer 
        lang = 'en'
        lang_dict = {'en':["Question", "Answer"],
                     'de':["Frage", "Antworten"],
                     'es':["Pregunta", "Responder"],
                     'vi':["câu hỏi", "trả lời"],
                     'zh':["问题", "答案"],
                     'ar':["\u0633\u0624\u0627\u0644","\u0625\u062C\u0627\u0628\u0647"],
                     'hi':["\u092A\u094D\u0930\u0936\u094D\u0928","\u091C\u0935\u093E\u092C"]}
        
        qword = lang_dict[lang][0]
        aword = lang_dict[lang][1]
        

        source_text = example.text_a
        target_text = example.meta["answer"].strip()
        question = example.meta["question"].strip()
        
        source_tokens = self.tokenizer.EncodeAsIds(" " + source_text.rstrip()).tokenization 
        question_tokens = self.tokenizer.EncodeAsIds(" " + qword + ": " + question).tokenization
        answer_prompt = self.tokenizer.EncodeAsIds(" " + aword + ": ").tokenization
        period_id = self.tokenizer.TokenToId('.')
        
        max_src_length = self.max_src_length - len(question_tokens) - len(answer_prompt) - 3 

        if max_src_length <= 0:
            print(question)

        assert max_src_length > 0
        source_tokens = [self.cls_id] + source_tokens[:max_src_length] + [period_id] + question_tokens + answer_prompt + [self.mask_id]

        return source_tokens, target_text

class QuesGenPVP(PVP):
    @property
    def mask_id(self):
        mask_token = 'sMASK' if self.task_mask else 'MASK'
        return self.tokenizer.get_command(mask_token).Id

    def encode(self, example: InputExample):
        source_text = example.text_a
        target_text, answer = example.meta["question"], example.meta["answer"]
        source_tokens = self.tokenizer.EncodeAsIds(source_text.rstrip() + " Question:").tokenization
        answer_tokens = self.tokenizer.EncodeAsIds(" Answer: " + answer).tokenization
        if len(source_tokens) > self.max_src_length - len(answer_tokens) - 2:
            max_src_length = self.max_src_length - len(answer_tokens) - 2
            answer_pattern = self.tokenizer.EncodeAsIds(" " + answer).tokenization

            def sub_finder(mylist, pattern):
                matches = []
                for i in range(len(mylist)):
                    if mylist[i] == pattern[0] and mylist[i:i + len(pattern)] == pattern:
                        matches.append(i)
                return matches

            answer_indices = sub_finder(source_tokens, answer_pattern)
            if len(answer_indices) == 0:
                print(f"Answer {answer} not exists in the source text")
                source_tokens = source_tokens[:max_src_length]
            else:
                start_index = max(answer_indices[0] - max_src_length // 2, 0)
                source_tokens = source_tokens[start_index: start_index + max_src_length]
        source_tokens = [self.cls_id] + source_tokens + [self.mask_id] + answer_tokens
        print(source_tokens)
        print(target_text)
        return source_tokens, target_text


class ChineseQAPVP(PVP):
    def encode(self, example: InputExample):
        source_text = example.text_a
        target_text = example.meta["answer"].strip()
        question = example.meta["question"].strip()
        source_tokens = self.tokenizer.EncodeAsIds(source_text.rstrip()).tokenization
        question_tokens = self.tokenizer.EncodeAsIds("问题：" + question + "答案：").tokenization
        max_src_length = self.max_src_length - len(question_tokens) - 2
        if max_src_length <= 0:
            print(question)
            question_tokens = question_tokens[self.max_src_length // 4]
        source_tokens = [self.cls_id] + question_tokens + [self.mask_id] + source_tokens[:max_src_length]
        return source_tokens, target_text




PVPS = {
    "gigaword": SummaryPVP,
    "cnn_dm": SummaryPVP,
    "cnn_dm_original": SummaryPVP,
    "xsum": SummaryPVP,
    "xlsum": SummaryPVP,
    "xwikis": SummaryPVP,
    "wikilingua": SummaryPVP,
    "scitldr": SummaryPVP,
    "ensum": SummaryPVP,
    "ncls": SummaryPVP,
    "ncls_multitask": NCLSPVP,
    "squad_generation": QuesGenPVP,
    "cmrc": ChineseQAPVP,
    "lcsts": NCLSPVP,
    "mlqa":MLQAPVP
}
