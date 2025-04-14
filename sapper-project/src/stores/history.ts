import { writable } from 'svelte/store';

const history = []

const { subscribe, set, update } = writable([]);

const addState = step => update(history => {
  return [...history, Object.assign({}, step)]
})

const resetState = () => {
  set(history);
};

const selectState = index => update(history => {
  return [...history.slice(0, index+1)];
})

export default {
  subscribe,
  addState,
  selectState,
  resetState
}