// OpenMP is required..
// g++-4.8 -ow2v -fopenmp -std=c++0x -Ofast -march=native -funroll-loops  main.cc  -lpthread

#include "word2vec.h"
#include <iostream>
#include <initializer_list>
#include <Windows.h>

using Sentence = Word2Vec<std::string>::Sentence;
using SentenceP = Word2Vec<std::string>::SentenceP;
int n_workers = 4;


int accuracy(Word2Vec<std::string>& model, std::string questions, int restrict_vocab = 30000) {
	std::ifstream in(questions);
	std::string line;
	auto lower = [](std::string& data) { std::transform(data.begin(), data.end(), data.begin(), ::tolower); };
	size_t count = 0, correct = 0, ignore = 0, almost_correct = 0;
	const int topn = 10;
	while (std::getline(in, line)) {
		if (line[0] == ':') {
			printf("%s\n", line.c_str());
			continue;
		}

		std::istringstream iss(line);
		std::string a, b, c, expected;
		iss >> a >> b >> c >> expected;
		lower(a); lower(b); lower(c); lower(expected);

		if (!model.has(a) || !model.has(b) || !model.has(c) || !model.has(expected)) {
			printf("unhandled: %s %s %s %s\n", a.c_str(), b.c_str(), c.c_str(), expected.c_str());
			++ignore;
			continue;
		}

		++count;
		std::vector<std::string> positive{ b, c }, negative{ a };
		auto predict = model.most_similar(positive, negative, topn);
		if (predict[0].first == expected) { ++correct; ++almost_correct; }
		else {
			bool found = false;
			for (auto& v : predict) {
				if (v.first == expected) { found = true; break; }
			}
			if (found) ++almost_correct;
			else printf("predicted: %s, expected: %s\n", predict[0].first.c_str(), expected.c_str());
		}
	}

	if (count > 0) printf("predict %lu out of %lu (%f%%), almost correct %lu (%f%%) ignore %lu\n", correct, count, correct * 100.0 / count, almost_correct, almost_correct * 100.0 / count, ignore);

	return 0;
}



void testing(Word2Vec<std::string>& model)
{
	SetConsoleCP(1251);// установка кодовой страницы win-cp 1251 в поток ввода
	SetConsoleOutputCP(1251); // установка кодовой страницы win-cp 1251 в поток вывода

	auto distance = [&model]() {
		while (1) {

			std::string s;
			std::cout << "\nFind nearest word for (:quit to break):";
			std::cin >> s;
			if (s == ":quit") break;
			std::cout << s << std::endl;
			auto p = model.most_similar(std::vector<std::string>{s}, std::vector<std::string>(), 10);
			size_t i = 0;
			for (auto& v : p) {
				std::cout << i++ << " " << v.first << " " << v.second << std::endl;
			}
		}
	};

	auto cstart = std::chrono::high_resolution_clock::now();
	model.load("vectors.bin");
	auto cend = std::chrono::high_resolution_clock::now();
	printf("load model: %.4f seconds\n", std::chrono::duration_cast<std::chrono::microseconds>(cend - cstart).count() / 1000000.0);

	distance();

	cstart = cend;
	accuracy(model, "questions-words.txt");
	cend = std::chrono::high_resolution_clock::now();
	printf("test model: %.4f seconds\n", std::chrono::duration_cast<std::chrono::microseconds>(cend - cstart).count() / 1000000.0);

}

void training(Word2Vec<std::string>& model)
{
	std::vector<SentenceP> sentences;

	size_t count = 0;
	const size_t max_sentence_len = 200;

	SentenceP sentence(new Sentence);
	// wget http://mattmahoney.net/dc/text8.zip
	//здесь имя файла из которого берем слова для обучения
	std::ifstream in("text8");
	while (true) {
		std::string s;
		in >> s;
		if (s.empty()) break;

		++count;
		sentence->tokens_.push_back(std::move(s));
		if (count == max_sentence_len) {
			count = 0;
			sentence->words_.reserve(sentence->tokens_.size());
			sentences.push_back(std::move(sentence));
			sentence.reset(new Sentence);
		}
	}

	if (!sentence->tokens_.empty())
		sentences.push_back(std::move(sentence));

	auto cstart = std::chrono::high_resolution_clock::now();
	model.build_vocab(sentences);
	auto cend = std::chrono::high_resolution_clock::now();
	printf("load vocab: %.4f seconds\n", std::chrono::duration_cast<std::chrono::microseconds>(cend - cstart).count() / 1000000.0);

	cstart = cend;
	model.train(sentences, n_workers);
	cend = std::chrono::high_resolution_clock::now();
	printf("train: %.4f seconds\n", std::chrono::duration_cast<std::chrono::microseconds>(cend - cstart).count() / 1000000.0);

	cstart = cend;
	model.save("vectors.bin");
	model.save_text("vectors.txt");
	cend = std::chrono::high_resolution_clock::now();
	printf("save model: %.4f seconds\n", std::chrono::duration_cast<std::chrono::microseconds>(cend - cstart).count() / 1000000.0);

	//		distance();
}


int main(int argc, const char *argv[])
{
	//для адекватной кириллицы в консоли ставить шрифт надо в окне консоли
	SetConsoleCP(1251);// установка кодовой страницы win-cp 1251 в поток ввода
	SetConsoleOutputCP(1251); // установка кодовой страницы win-cp 1251 в поток вывода
	Word2Vec<std::string> model(200);


	model.sample_ = 0;
	//	model.window_ = 10;
	//	model.phrase_ = true;


	::srand(::time(NULL));

	bool train = false, test = true;//по умолчанию тест
	if (argc > 1 && std::string(argv[1]) == "train") {//если вызов с консоли с аргументом обучения - на всякий
		std::swap(train, test);
	}
	else
	{
		std::string s;//или через син введено(для тестов)
		std::cout << "train для обучения, другое - тест" << std::endl;
		std::cin >> s;
		if(s=="train")
			std::swap(train, test);
	}

	if (train) {
		training(model);
	}

	if (test) {
		testing(model);
	}

	return 0;
}

