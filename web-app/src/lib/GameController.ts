import { writable } from 'svelte/store';
import { tick } from 'svelte';
import { game } from '../game.js';
import { AgentFactory, type BaseAgent, type AgentType } from './Agent.js';
import history from '../stores/history';
import { get } from 'svelte/store';

function createGameStore() {
  // State
  const score = writable(game.getScore());
  const agent_play = writable(true);
  const outcome = writable<string | null>(null);
  const state = writable<number[]>([0, 0, 0, 0, 0, 0, 0, 0, 0]);
  const turn = writable<boolean>(true);
  const notification = writable('');
  const show_notification = writable(false);
  const agentLoading = writable(false);
  const agentPlayer = writable<'X' | 'O'>('O');
  const agentType = writable<AgentType>('TDAgent');
  const showHistory = writable(false);
  const showValueFunction = writable(true);

  const availableAgentTypes = AgentFactory.getAvailableAgentTypes();

  // Non-reactive
  let agentX: BaseAgent | null = null;
  let agentO: BaseAgent | null = null;
  let agent: BaseAgent | null = null; // The agent that plays (O or X)
  let timeoutId: number | null = null;

  // Methods
  async function updateState() {
    const $history = get(history);
    if ($history.length > 0) {
      state.set($history[$history.length - 1].state);
      turn.set($history[$history.length - 1].turn);
    }
  }

  async function selectMarker(index: number, isAgentMove = false) {
    const $agent_play = get(agent_play);
    const $turn = get(turn);
    const $agentPlayer = get(agentPlayer);
    if ($agent_play && isAgentTurn($turn, $agentPlayer) && !isAgentMove) return;

    const result = game.updateState(index);
    outcome.set(result);
    score.set(game.getScore());

    if (result == 'win') {
      history.addState({ state: [...game.getStates()], turn: game.getTurn() });
      await tick();
      updateState();
      showNotification(winnerInfo(game.getWinner() ?? 0));
      reset();
    } else if (result == 'tie') {
      showNotification('Tie');
      reset();
    } else if (result == 'invalid') {
      showNotification('Impossible Move');
    } else {
      history.addState({ state: [...game.getStates()], turn: game.getTurn() });
      await tick();
      updateState();
      if ($agent_play && isAgentTurn(game.getTurn(), $agentPlayer) && agent && !isAgentMove) {
        setTimeout(() => makeAgentMove(), 500);
      }
    }
  }

  function isAgentTurn(turn: boolean, agentPlayer: 'X' | 'O') {
    return agentPlayer === 'X' ? turn === true : turn === false;
  }

  async function makeAgentMove() {
    const $turn = get(turn);
    const $agentPlayer = get(agentPlayer);
    if (!agent || !isAgentTurn($turn, $agentPlayer)) return;
    try {
      const currentState = game.getStates();
      const agentMove = agent.chooseAction(currentState);
      await selectMarker(agentMove, true);
    } catch {
      showNotification('Agent Error');
    }
  }

  function showNotification(msg: string) {
    notification.set(msg);
    show_notification.set(true);
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
    timeoutId = setTimeout(() => {
      show_notification.set(false);
      notification.set('');
    }, 2000);
  }

  function winnerInfo(winner: number) {
    if (winner == 1) return 'Player 1 Wins';
    else if (game.agentPlaying()) return 'Agent Wins';
    else return 'Player 2 Wins';
  }

  async function reset(timeout = 1500) {
    /*
      Reset the game
      - If there is a timeout, clear it
      - Set the timeout to 1.5 seconds
      - Reset the game
      - Reset the history
    */
    if (timeoutId) clearTimeout(timeoutId);
    timeoutId = setTimeout(async () => {
      game.reset();
      history.resetState();
      history.addState({ state: [...game.getStates()], turn: game.getTurn() });
      await tick();
      updateState();
      let $agent_play, $agentPlayer;
      agent_play.subscribe(v => $agent_play = v)();
      agentPlayer.subscribe(v => $agentPlayer = v)();
      if ($agent_play && $agentPlayer === 'X' && agent) {
        setTimeout(() => makeAgentMove(), 500);
      }
    }, timeout);
  }

  async function loadAgent() {
    const $agentType = get(agentType);
    agentLoading.set(true);
    try {
      agentX = await AgentFactory.loadAgent($agentType, 'X');
      agentO = await AgentFactory.loadAgent($agentType, 'O');
      // The agent that plays (for moves) is still based on agentPlayer
      const $agentPlayer = get(agentPlayer);
      agent = $agentPlayer === 'X' ? agentX : agentO;
      game.playAgent();
      showNotification(`${$agentType} loaded!`);
      await reset(0);
    } catch {
      showNotification('Failed to Load Agent');
    }
    agentLoading.set(false);
  }

  async function toggleAgentPlay() {
    /*
      Toggle the agent play
      - If the agent is not loaded and the agent is not playing, load the agent
      - Otherwise, toggle the agent play
    */
    const $agent_play = get(agent_play);
    const $agentType = get(agentType);
    const $agentPlayer = get(agentPlayer);
    if (!$agent_play && !agent) {
      agentLoading.set(true);
      try {
        agent = await AgentFactory.loadAgent($agentType, $agentPlayer);
        game.playAgent();
        agent_play.set(true);
        showNotification(`${$agentType} ${$agentPlayer} Loaded!`);
        await reset(0);
      } catch {
        showNotification('Failed to Load Agent');
      }
      agentLoading.set(false);
    } else {
      agent_play.set(!$agent_play);
      if (!$agent_play) game.playAgent();
      await reset(0);
    }
  }

  function getValueForCell(index: number): number | null {
    const $turn = get(turn);
    // Use the value function for the player whose turn it is
    const vfAgent = $turn ? agentX : agentO;
    if (!vfAgent) return null;
    const currentState = game.getStates();
    if (currentState[index] !== 0) return null;
    const player = $turn ? 1 : 2;
    const nextState = [...currentState];
    nextState[index] = player;

    // Type guards
    const isTDAgent = (a: any): a is { getValue: (stateKey: string) => number, getStateKey: (state: number[]) => string } =>
      typeof a.getValue === 'function' && typeof a.getStateKey === 'function' && (a.constructor.name === 'TDAgent' || a.constructor.name === 'MinMaxTDAgent');
    const isQLearningAgent = (a: any): a is { getActionValue: (stateKey: string, action: number) => number, getStateKey: (state: number[]) => string } =>
      typeof a.getActionValue === 'function' && typeof a.getStateKey === 'function';

    try {
      if (isTDAgent(vfAgent)) {
        const stateKey = vfAgent.getStateKey(nextState);
        return vfAgent.getValue(stateKey);
      } else if (isQLearningAgent(vfAgent)) {
        const stateKey = vfAgent.getStateKey(currentState);
        return vfAgent.getActionValue(stateKey, index);
      }
      return null;
    } catch {
      return null;
    }
  }

  // Return the store API
  return {
    score,
    agent_play,
    outcome,
    state,
    turn,
    notification,
    show_notification,
    agentLoading,
    agentPlayer,
    agentType,
    showHistory,
    showValueFunction,
    availableAgentTypes,
    updateState,
    selectMarker,
    makeAgentMove,
    reset,
    toggleAgentPlay,
    loadAgent,
    getValueForCell
  };
}

export const gameStore = createGameStore(); 