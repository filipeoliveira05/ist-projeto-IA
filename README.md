# 🧩 Projeto NURUOMINO – Inteligência Artificial 2024/25

## 📌 Sobre o Projeto
Este repositório contém a implementação do projeto **NURUOMINO** desenvolvido para a unidade curricular **Inteligência Artificial (IA)** do curso de Engenharia Informática e de Computadores (IST).  
O objetivo é resolver uma adaptação do puzzle **NURUOMINO (LITS)** utilizando **técnicas de procura** e princípios de IA. A solução é implementada em **Python** e determina a disposição correta das peças na grelha, respeitando todas as regras do puzzle.

---

## 🎯 Objetivos
- Implementar um **resolvedor do puzzle NURUOMINO** em Python.
- Modelar o problema com **estados**, **ações**, **resultados** e **teste de objetivo**.
- Integrar **algoritmos de procura** (não informada e informada): DFS, BFS, Greedy e A*.
- Desenvolver uma **heurística eficiente** para guiar a procura informada.
- Garantir as regras do puzzle:
  - Cada **região** contém exatamente **um tetraminó** (L, I, T, S).
  - **Tetrominos iguais não podem ficar ortogonalmente adjacentes** (rotações/reflexões contam como iguais).
  - Todas as células preenchidas formam um **nurikabe válido**: conectadas ortogonalmente e **sem blocos 2×2**.

---

## 🛠 O que foi feito
- **Modelação do problema** na classe `Nuruomino`.
- **Classe `Board`**: representação interna da grelha e métodos auxiliares (e.g., regiões adjacentes, valores adjacentes, impressão).
- **Classe `NuruominoState`**: representação de estados com identificadores únicos e comparação para desempate.
- **Funções chave**:
  - `parse_instance()` – leitura da instância a partir do **stdin**.
  - `actions(state)` – geração de ações a partir de um estado.
  - `result(state, action)` – aplicação de uma ação e obtenção do novo estado.
  - `goal_test(state)` – verificação de estado objetivo.
  - `h(node)` – **heurística** para A* / Greedy.
- **Integração com algoritmos de procura** (`search.py`): `depth_first_tree_search`, `breadth_first_tree_search`, `greedy_search`, `astar_search`.
- **Suporte a testes**:
  - Pasta **`public/`** com instâncias de teste.
  - Script **`test_runner.py`** para executar os testes de forma **rápida** e **automatizada**.

---

## 📂 Estrutura do Repositório
- [nuruomino.py](./nuruomino.py) → Implementação principal
- [search.py](./search.py) → Código base de procura
- [utils.py](./utils.py) → Utilitários de procura
- [public/](./public/) → Instâncias de teste (ficheiros de input)
- [test_runner.py](./test_runner.py) → Script Python para correr os testes de forma eficiente
- [enunciado-IA2425.pdf](./enunciado-IA2425.pdf) → Documento com as regras e especificações do projeto.

---

## ✅ Formato de Input e Output
**Input** (lido do `stdin`): grelha `N×N` com identificadores de região (strings de números), p.ex.:
```
<region-number-l1c1> ... <region-number-l1cN>
...
<region-number-lNc1> ... <region-number-lNcN>
```

**Exemplo de input:**
```
1 1 2 2 3 3
1 2 2 2 3 3
1 3 3 2 3 5
3 3 3 3 3 5
4 4 4 3 3 5
4 3 3 3 3 5
```

**Output**: grelha solução, com letras das peças nas regiões preenchidas (L, I, T, S), p.ex.:
```
L L S 2 3 3
L 2 S S 3 3
L 3 3 S 3 I
3 3 3 T 3 I
L L L T T I
L 3 3 T 3 I
```

---

## ▶️ Como Executar
**Resolver uma instância única:**
```bash
python3 nuruomino.py < public/test01.txt
```

**Correr todos os testes com o script auxiliar:**
```bash
python3 test_runner.py
```
O `test_runner.py` percorre os ficheiros em `public/`, executa o resolvedor com *stdin* redirecionado, e pode apresentar métricas como **tempo de execução** e **validação do formato de output**.

---

## 🧪 Testes
- As **instâncias** estão em `public/`.
- O **`test_runner.py`** foi criado para:
  - Executar automaticamente todas as instâncias disponíveis.
  - Validar o **formato** do output e a **terminação** do algoritmo.
  - (Opcional) Medir e comparar **tempos** entre estratégias de procura.
- Exemplos de uso:
  ```bash
  # Executar apenas instâncias que começam por "test0"
  python3 test_runner.py --pattern "public/test0*.txt"

  # Executar e guardar resultados numa pasta
  python3 test_runner.py --out-dir results/
  ```

---

## 🔧 Requisitos
- **Python 3.10+** (recomendado 3.10, 3.11 ou 3.12)
- **NumPy** (opcional) para operações matriciais:
  ```bash
  pip install numpy
  ```

---

## 🔗 Referências e Links Úteis
- Puzzle **LITS** (NURUOMINO): <https://en.wikipedia.org/wiki/LITS>
- Código base AIMA (adaptado): <https://github.com/aimacode>
- Vídeo explicativo do puzzle: <https://youtu.be/q3mspJmENnQ?si=JrczxUu1gDOBiV4Q>

---

## 📝 Notas Finais
- Os ficheiros `search.py` e `utils.py` **não devem ser alterados**; quaisquer alterações necessárias devem ser feitas no `nuruomino.py`.
- O código segue o **formato de input/output** indicado para facilitar a correção automática.
- O repositório inclui **`public/`** e **`test_runner.py`** para tornar o desenvolvimento e a validação mais rápidos e reprodutíveis.
