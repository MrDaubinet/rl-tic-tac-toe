import { writable } from 'svelte/store';

interface GameState {
  state: number[];
  turn: boolean;
}

const history: GameState[] = []

const { subscribe, set, update } = writable<GameState[]>([]);

const addState = (step: GameState) => update(history => {
  return [...history, Object.assign({}, step)]
})

const resetState = () => {
  set([]);
};

const selectState = (index: number) => update(history => {
  return [...history.slice(0, index+1)];
})

export default {
  subscribe,
  addState,
  selectState,
  resetState
} 