#!/bin/bash
# analyze_profile.sh
# Script d'analyse automatisée des résultats de profiling

set -e

BUILD_DIR="build"
RESULTS="results"
RESULTS_DIR="${RESULTS}/profiling_results"
PERF_DATA="${BUILD_DIR}/perf.data"
TEST_DIR="${BUILD_DIR}/test/"

# Couleurs
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}   ANALYSE PROFILING CPU - HMM${NC}"
echo -e "${BLUE}============================================${NC}\n"

# Vérifier que perf.data existe
if [ ! -f "$PERF_DATA" ]; then
    echo -e "${RED}Erreur: $PERF_DATA introuvable${NC}"
    echo -e "${YELLOW}Exécutez d'abord: make profile-cpu${NC}"
    exit 1
fi

# Créer répertoire résultats
mkdir -p "$RESULTS_DIR"

# ==============================================================================
# 1. RAPPORT STANDARD PERF
# ==============================================================================
echo -e "${BLUE}[1/5] Génération rapport perf standard...${NC}"
cd "$BUILD_DIR"
perf report -i perf.data -n --stdio > "${RESULTS_DIR}/perf_report_full.txt"
echo -e "${GREEN}✓ Rapport complet: ${RESULTS_DIR}/perf_report_full.txt${NC}\n"

# ==============================================================================
# 2. TOP FONCTIONS COÛTEUSES
# ==============================================================================
echo -e "${BLUE}[2/5] Extraction top 20 fonctions...${NC}"
perf report -i perf.data -n --stdio --sort=overhead | head -n 50 > "${RESULTS_DIR}/top_functions.txt"
echo -e "${GREEN}✓ Top fonctions: ${RESULTS_DIR}/top_functions.txt${NC}\n"

# ==============================================================================
# 3. CALL GRAPH PAR ALGO
# ==============================================================================
echo -e "${BLUE}[3/5] Génération call graphs...${NC}"

# Forward
perf report -i perf.data -g --stdio --sort=comm | \
    grep -A 100 "forward_algorithm_log" | head -n 100 > "${RESULTS_DIR}/callgraph_forward.txt" 2>/dev/null || true

# Backward
perf report -i perf.data -g --stdio --sort=comm | \
    grep -A 100 "backward_algorithm_log" | head -n 100 > "${RESULTS_DIR}/callgraph_backward.txt" 2>/dev/null || true

# Cholesky
perf report -i perf.data -g --stdio --sort=comm | \
    grep -A 100 "choleskyDecomposition" | head -n 100 > "${RESULTS_DIR}/callgraph_cholesky.txt" 2>/dev/null || true

echo -e "${GREEN}✓ Call graphs générés${NC}\n"

# ==============================================================================
# 4. STATISTIQUES CACHE/BRANCHES
# ==============================================================================
echo -e "${BLUE}[4/5] Analyse cache et branches (si disponible)...${NC}"
perf stat -d ./test_profile_cpu > "${RESULTS_DIR}/perf_stat.txt" 2>&1 || \
    echo -e "${YELLOW}⚠ perf stat non disponible (nécessite droits root)${NC}"

# ==============================================================================
# 5. GÉNÉRATION FLAMEGRAPH (SI DISPONIBLE)
# ==============================================================================
echo -e "${BLUE}[5/5] Génération flamegraph...${NC}"

FLAMEGRAPH_DIR="../../scripts/FlameGraph"
if [ -d "$FLAMEGRAPH_DIR" ]; then
    perf script -i perf.data | \
        ${FLAMEGRAPH_DIR}/stackcollapse-perf.pl | \
        ${FLAMEGRAPH_DIR}/flamegraph.pl > "${RESULTS_DIR}/flamegraph.svg"
    echo -e "${GREEN}✓ Flamegraph: ${RESULTS_DIR}/flamegraph.svg${NC}\n"
else
    echo -e "${YELLOW}⚠ FlameGraph tools non trouvés (optionnel)${NC}"
    echo -e "${YELLOW}  Clone depuis: git clone https://github.com/brendangregg/FlameGraph scripts/FlameGraph${NC}\n"
fi

# ==============================================================================
# RÉSUMÉ
# ==============================================================================
cd ..
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}   ANALYSE TERMINÉE${NC}"
echo -e "${GREEN}============================================${NC}\n"

echo "Fichiers générés dans ${RESULTS}/profiling_results/ :"
ls -lh "${RESULTS_DIR}/" | tail -n +2 | awk '{print "  - " $9 " (" $5 ")"}'

echo -e "\n${BLUE}Commandes utiles :${NC}"
echo -e "  ${YELLOW}Voir top fonctions :${NC}"
echo "    cat ${RESULTS_DIR}/top_functions.txt | head -n 30"
echo -e "\n  ${YELLOW}Voir call graph forward :${NC}"
echo "    cat ${RESULTS_DIR}/callgraph_forward.txt"
echo -e "\n  ${YELLOW}Visualiser flamegraph :${NC}"
echo "    firefox ${RESULTS_DIR}/flamegraph.svg"
echo -e "\n  ${YELLOW}Rapport interactif :${NC}"
echo "    cd ${BUILD_DIR} && perf report -i perf.data"

echo ""